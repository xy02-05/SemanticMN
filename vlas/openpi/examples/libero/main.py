import collections
import dataclasses
import logging
import math
import pathlib
import random

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import torch
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    # Action feature 存储开关：开启后在推理时提取每个 action chunk 的 mean pool 特征
    save_action_features: bool = False
    feature_save_dir: str = ""  # 为空时默认存到 video_out_path 同级的 action_features 目录


def eval_libero(args: Args) -> None:
    _set_eval_seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # feature 存储目录：默认在 video_out_path 同级的 action_features
    feat_dir = None
    if args.save_action_features:
        feat_dir = pathlib.Path(args.feature_save_dir) if args.feature_save_dir else \
            pathlib.Path(args.video_out_path).parent / "action_features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Action feature saving enabled → {feat_dir}")

    # 全 suite 汇总容器（仅在开启 feature 时使用）
    all_traj_features = []  # 每条轨迹的 mean pool feature
    all_traj_meta = []      # (task_id, episode_idx, success, task_desc, n_chunks)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # 每个episode独立seed环境，防止reset()消耗的np.random状态跨episode累积
            env.seed(_episode_env_seed(args.seed, task_id, episode_idx))
            env.reset()
            action_plan = collections.deque()
            replan_count = 0

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            # 当前 episode 的 chunk features 收集器
            episode_chunk_features = [] if args.save_action_features else None

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # 每次replan生成确定性noise_seed，基于(seed,task,episode,replan次数)
                        # 这使得不同num_trials_per_task的评测在相同(task,episode)上结果一致
                        noise_seed = _compute_noise_seed(args.seed, task_id, episode_idx, replan_count)
                        replan_count += 1

                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "noise_seed": np.int64(noise_seed),
                        }
                        # 注入 feature 提取请求标志
                        if args.save_action_features:
                            element["_extract_action_features"] = True

                        # Query model to get action (+ optional feature)
                        result = client.infer(element)
                        action_chunk = result["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                        # 收集 chunk feature: 服务端返回的 [L, D] mean pool 特征
                        if args.save_action_features and "action_feature" in result:
                            episode_chunk_features.append(result["action_feature"])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # 保存当前 episode 的 action features
            if args.save_action_features and episode_chunk_features:
                chunk_feats = np.stack(episode_chunk_features, axis=0)  # [N_chunks, L, D]
                traj_feat = chunk_feats.mean(axis=0)                    # [L, D]
                all_traj_features.append(traj_feat)
                all_traj_meta.append((task_id, episode_idx, bool(done), task_description, len(episode_chunk_features)))

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            # 用 task_id 和 episode_idx 组成唯一文件名，避免同一任务下 success/failure 视频被覆盖。
            rollout_name = f"rollout_task{task_id:02d}_episode{episode_idx + 1:03d}_{task_segment}_{suffix}.mp4"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / rollout_name,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # 保存全 suite 的轨迹级 feature 汇总
    if args.save_action_features and all_traj_features:
        _save_trajectory_summary(feat_dir, all_traj_features, all_traj_meta)


def _save_trajectory_summary(feat_dir, all_traj_features, all_traj_meta):
    """保存全 suite 的轨迹级 feature 汇总到 trajectory_summary.npz"""
    # 与 analysis/openpi_representation 使用相同层索引
    layer_indices = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 17])
    traj_features = np.stack(all_traj_features, axis=0)  # [N_trajs, L, D]
    task_ids = np.array([m[0] for m in all_traj_meta], dtype=np.int64)
    episode_ids = np.array([m[1] for m in all_traj_meta], dtype=np.int64)
    success = np.array([m[2] for m in all_traj_meta], dtype=bool)
    task_descs = np.array([m[3] for m in all_traj_meta])
    n_chunks = np.array([m[4] for m in all_traj_meta], dtype=np.int64)

    out_path = feat_dir / "trajectory_summary.npz"
    np.savez_compressed(
        out_path,
        traj_features=traj_features,
        task_ids=task_ids,
        episode_ids=episode_ids,
        success=success,
        task_descs=task_descs,
        n_chunks=n_chunks,
        layer_indices=layer_indices,
    )
    logging.info(f"Saved trajectory summary: {out_path} | shape={traj_features.shape} | "
                 f"success_rate={success.mean():.3f}")


def _episode_env_seed(base_seed: int, task_id: int, episode_idx: int) -> int:
    """为每个 (task, episode) 生成唯一的环境seed，使env.reset()的随机行为独立于其他episode。"""
    return base_seed + task_id * 100003 + episode_idx * 1003


def _compute_noise_seed(base_seed: int, task_id: int, episode_idx: int, replan_count: int) -> int:
    """为每次推理生成唯一的noise seed，使扩散噪声只取决于(seed, task, episode, replan步)。
    用质数乘子避免不同参数组合产生哈希碰撞。"""
    return base_seed + task_id * 1000003 + episode_idx * 10007 + replan_count * 13


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _set_eval_seed(seed: int) -> None:
    """客户端也固定常见随机源，避免 benchmark / torch.load 等路径引入额外漂移。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
