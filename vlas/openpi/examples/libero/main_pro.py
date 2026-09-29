import collections
import dataclasses
import logging
import math
import pathlib
import random
import sys

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import torch
import tqdm
import tyro

OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from examples.libero import libero_runtime


def _episode_env_seed(base_seed: int, task_id: int, episode_idx: int) -> int:
    """为每个 (task, episode) 生成唯一的环境seed。"""
    return base_seed + task_id * 100003 + episode_idx * 1003


def _compute_noise_seed(base_seed: int, task_id: int, episode_idx: int, replan_count: int) -> int:
    """为每次推理生成唯一的noise seed。"""
    return base_seed + task_id * 1000003 + episode_idx * 10007 + replan_count * 13

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
WOODEN_CABINET_RANGE = 0.16  # max slide travel from XML


def _log_articulated_joints(env, task_description):
    """Log joint states for cabinet/drawer tasks to diagnose which drawer moved."""
    keywords = ("cabinet", "drawer")
    if not any(kw in task_description.lower() for kw in keywords):
        return
    try:
        sim = env.env.sim if hasattr(env, "env") else env.sim
        model = sim.model
        try:
            all_names = [model.joint_id2name(i) for i in range(model.njnt)]
        except AttributeError:
            all_names = [model.joint(i).name for i in range(model.njnt)]

        drawer_joints = [n for n in all_names if n and "level" in n]
        if not drawer_joints:
            return

        logging.info("  --- Drawer joint states (end of episode) ---")
        for jname in drawer_joints:
            qpos_addr = model.get_joint_qpos_addr(jname)
            qpos = float(sim.data.qpos[qpos_addr])
            pct = abs(qpos) / WOODEN_CABINET_RANGE * 100 if qpos < 0 else 0.0
            is_open = qpos < -0.08
            if "top" in jname:
                label = "top"
            elif "middle" in jname:
                label = "MIDDLE"
            elif "bottom" in jname:
                label = "bottom"
            else:
                label = jname
            marker = " <<<TARGET" if "middle" in jname else ""
            logging.info(
                f"    [{label:>6s}] qpos={qpos:+.4f}  opened={pct:5.1f}%%  is_open(>50%%)={is_open}{marker}"
            )
    except Exception as e:
        logging.warning(f"  [drawer-debug] Could not read joints: {e}")
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
LIBERO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


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
    # LIBERO-PRO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial_lan"  # Task suite. Options also include LIBERO-PRO suffix suites, e.g. libero_spatial_task.
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int | None = None  # demo 模式下只跑前 N 个 task，先打通链路再扩大全量评测。

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

    runtime = libero_runtime.prepare_libero_runtime(
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        seed=args.seed,
    )
    if runtime.package_root is not None:
        logging.info("Using LIBERO package root: %s", runtime.package_root)

    benchmark_dict = runtime.benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    if args.max_tasks is not None:
        num_tasks_in_suite = min(num_tasks_in_suite, args.max_tasks)
        logging.info("Demo mode enabled, only evaluating the first %d tasks.", num_tasks_in_suite)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # feature 存储目录
    feat_dir = None
    if args.save_action_features:
        feat_dir = pathlib.Path(args.feature_save_dir) if args.feature_save_dir else \
            pathlib.Path(args.video_out_path).parent / "action_features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Action feature saving enabled → {feat_dir}")

    # 全 suite 汇总容器
    all_traj_features = []
    all_traj_meta = []

    base_suite_name = libero_runtime.get_base_suite_name(args.task_suite_name)
    if base_suite_name not in LIBERO_MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = LIBERO_MAX_STEPS[base_suite_name]

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task=task,
            resolution=LIBERO_ENV_RESOLUTION,
            seed=args.seed,
            get_libero_path=runtime.get_libero_path,
            offscreen_env_cls=runtime.offscreen_env_cls,
        )

        # 仅在文件名推导的 language 与 BDDL 中的不一致时打 log，方便确认扰动生效
        if task_description != task.language:
            logging.info(f"[PRO] BDDL language: {task_description!r}  (filename: {task.language!r})")

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            # 每个episode独立seed环境
            env.seed(_episode_env_seed(args.seed, task_id, episode_idx))
            env.reset()
            action_plan = collections.deque()
            replan_count = 0
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            episode_chunk_features = [] if args.save_action_features else None

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        noise_seed = _compute_noise_seed(args.seed, task_id, episode_idx, replan_count)
                        replan_count += 1

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
                        if args.save_action_features:
                            element["_extract_action_features"] = True

                        result = client.infer(element)
                        action_chunk = result["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                        if args.save_action_features and "action_feature" in result:
                            episode_chunk_features.append(result["action_feature"])

                    action = action_plan.popleft()
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

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            rollout_name = f"rollout_task{task_id:02d}_episode{episode_idx + 1:03d}_{task_segment}_{suffix}.mp4"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / rollout_name,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # 保存全 suite 的轨迹级 feature 汇总
    if args.save_action_features and all_traj_features:
        _save_trajectory_summary(feat_dir, all_traj_features, all_traj_meta)


def _save_trajectory_summary(feat_dir, all_traj_features, all_traj_meta):
    """保存全 suite 的轨迹级 feature 汇总到 trajectory_summary.npz"""
    layer_indices = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 17])
    traj_features = np.stack(all_traj_features, axis=0)
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


def _parse_bddl_language(bddl_path: pathlib.Path) -> str | None:
    """从 BDDL 文件中提取 (:language ...) 字段。
    LIBERO-PRO 的扰动只改 BDDL 内容，不改文件名，
    所以必须从文件内容读 language 才能拿到扰动后的 task description。"""
    import re
    text = bddl_path.read_text(encoding="utf-8")
    m = re.search(r'\(:language\s+(.+?)\)', text)
    return m.group(1).strip() if m else None


def _get_libero_env(task, resolution, seed, get_libero_path, offscreen_env_cls):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # 优先从 BDDL 文件内容读取 language（支持 LIBERO-PRO 扰动），
    # 若读取失败则 fallback 到文件名推导的 task.language
    task_description = _parse_bddl_language(task_bddl_file) or task.language
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = offscreen_env_cls(**env_args)
    env.seed(seed)
    return env, task_description


def _set_eval_seed(seed: int) -> None:
    """客户端也固定常见随机源，避免 benchmark / torch.load 等路径引入额外漂移。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
