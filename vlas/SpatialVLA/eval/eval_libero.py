"""
SpatialVLA × LIBERO 评测主入口
================================
设计原则：
  1. 完全独立——不导入任何 vlas/openpi/* 代码；libero 包从本目录同级 LIBERO/ 注入 sys.path，
     与 openpi 评测使用的全局 libero 隔离。
  2. 模型加载用 transformers.AutoModel + AutoProcessor + trust_remote_code=True，
     直接读 ckpt 目录（含 model.safetensors / processor configs / dataset_statistics 等）。
  3. 与训练对齐：center crop crop_pct=0.9（训练 random_resized_crop scale=0.9 的等价确定性版本）；
     LIBERO sim 图像逆时针旋转 180°（与 OpenVLA modified_libero_rlds 数据格式一致）；
     gripper 方向反转（训练 transform 把 LIBERO -1..1 → 0..1 invert）。
  4. CogACT 风格 AdaptiveEnsembler（cosine-similarity 加权），默认 alpha=0.1；
     可关闭走原始 chunk[0] 路径。

参考代码（已 work 的 evaluation 模板）：
  - openpi/examples/libero/main.py（LIBERO env 启动 + episode 主循环手法；本脚本独立重写）
  - SpatialVLA/test/test_huggingface.py（model API 用法）
  - microsoft/CogACT/sim_cogact/adaptive_ensemble.py（AdaptiveEnsembler 算法）
  - openvla/experiments/robot/libero/run_libero_eval.py（max_steps per suite 数值）
"""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
import random
import sys
from typing import List, Optional

import imageio
import numpy as np
import torch
import tqdm
import tyro
from PIL import Image

# ============ 关键：隔离 LIBERO import ============
# 我们使用 vlas/SpatialVLA/LIBERO/ 这一份（单独 cp 自 projects/LIBERO-official），
# 与 openpi 评测可能依赖的 site-packages 版 libero 完全隔离。
# 本文件 = vlas/SpatialVLA/eval/eval_libero.py，因此 LIBERO/ = ../LIBERO
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_SPATIALVLA_ROOT = _THIS_DIR.parent
_LIBERO_LOCAL = _SPATIALVLA_ROOT / "LIBERO"
if _LIBERO_LOCAL.exists() and str(_LIBERO_LOCAL) not in sys.path:
    sys.path.insert(0, str(_LIBERO_LOCAL))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

# 同目录的 ensembler 与 launcher 之间不必再做 sys.path 处理；脚本启动时
# 由 launcher 把 vlas/SpatialVLA/eval 加入 PYTHONPATH 即可
sys.path.insert(0, str(_THIS_DIR))
from action_ensembler import AdaptiveEnsembler  # noqa: E402

# ============ 常量 ============
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # 等待物体 settle 期间发空动作
LIBERO_ENV_RESOLUTION = 256  # LIBERO 训练数据 + sim 默认分辨率

# 各 suite 的最大 step（来自 OpenVLA / openpi 实测；值取自 longest demo + 适当余量）
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # ============ 模型 ============
    ckpt_path: str  # SpatialVLA ckpt 目录（含 model.safetensors + processor configs）
    unnorm_key: str = "libero_mix_no_noops/1.0.0"  # joint train 4 suite 共用同一份 norm_stats
    dtype: str = "bf16"  # bf16 / fp32
    attn_implementation: str = "eager"  # ZoeDepth 不支持 sdpa；eager 兼容性最好

    # ============ 评测范围 ============
    suites: str = "libero_spatial,libero_object,libero_goal,libero_10"  # 逗号分隔
    num_trials_per_task: int = 10  # 与用户约定一致；可命令行覆盖
    max_tasks_per_suite: int = -1  # -1 表示 suite 内全部 task；smoke test 时设 1 加速
    num_steps_wait: int = 10  # 让物体落下 settle

    # ============ Image / prompt 对齐 ============
    use_center_crop: bool = True
    crop_pct: float = 0.9  # 训练 random_resized_crop scale=0.9 的对应 center crop
    rotate_image_180: bool = True  # modified_libero_rlds 创建时已 rotate 180°（OpenVLA regenerate 注释明示）；sim env 输出未 rotate，eval 必须旋转匹配训练分布
    prompt_template: str = "What action should the robot take to {task}?"

    # ============ Action 后处理 ============
    use_action_ensemble: bool = True
    action_ensemble_alpha: float = 0.1  # CogACT AAE 默认；alpha=0 等价于均权
    replan_steps: int = 4  # 用满 chunk_size=4；ensemble 开启时该值不影响输出（每步都 ensemble）
    action_clip: bool = True  # clip 到 [-1, 1] 防极端值
    invert_gripper: bool = True  # SpatialVLA decode 出的 gripper(0=open,1=close) → LIBERO(-1=open,+1=close)
    gripper_threshold: float = 0.5  # >threshold 才视为 close（去 jitter）

    # ============ 输出 ============
    video_out_dir: str = "outputs/eval_libero"  # 相对 spatialvla 根目录
    save_videos: bool = True
    video_fps: int = 10

    # ============ 其它 ============
    seed: int = 7
    deterministic: bool = True  # cudnn 确定性（跨次结果可比）


# =================================================================
# 辅助：seed / quaternion 转换 / image preprocessing
# =================================================================
def _set_seed_everywhere(seed: int, deterministic: bool) -> None:
    """固定常见随机源；deterministic=True 时 cudnn 也固定，跨次结果一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _episode_seed(base: int, task_id: int, ep_idx: int) -> int:
    """为每个 (task, episode) 生成唯一 env seed，避免跨 episode 状态污染。"""
    return base + task_id * 100003 + ep_idx * 1003


def _center_crop_then_resize(pil_img: Image.Image, crop_pct: float, target_size: int) -> Image.Image:
    """
    与训练 random_resized_crop scale=0.9 对齐的确定性版本：
    取中心 crop_pct 比例的方块，再 resize 到 target_size。
    """
    W, H = pil_img.size
    cw, ch = int(W * crop_pct), int(H * crop_pct)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    return pil_img.crop((x0, y0, x0 + cw, y0 + ch)).resize((target_size, target_size))


def _preprocess_obs_image(obs_img: np.ndarray, args: Args) -> Image.Image:
    """
    LIBERO env 输出的 obs['agentview_image'] → 喂给 SpatialVLA processor 的 PIL：
      1. （可选）逆时针 180° 旋转：OpenVLA modified_libero_rlds 训练数据是反转过的，
         sim env 是正向，故 eval 必须把 sim 图像翻成训练那样才能匹配。
      2. （可选）center crop scale=0.9：与训练 random_resized_crop 对齐。
      3. 转 PIL（processor 自己 resize 到 model 期望的 224×224）。
    """
    img = obs_img
    if args.rotate_image_180:
        # 上下 + 左右翻转 ≡ 旋转 180°（连续切片以避免 negative-stride 报错）
        img = np.ascontiguousarray(img[::-1, ::-1])
    pil = Image.fromarray(img.astype(np.uint8))
    if args.use_center_crop:
        pil = _center_crop_then_resize(pil, args.crop_pct, target_size=LIBERO_ENV_RESOLUTION)
    return pil


def _postprocess_action(action: np.ndarray, args: Args) -> np.ndarray:
    """
    Decode 出的 7-dim action → LIBERO env step 接收的 7-dim action：
      - clip 到 [-1, 1]（可选，防 NaN/极端值）
      - gripper 方向修正（**关键**）：
          训练 pipeline (libero_dataset_transform) 处理：
            LIBERO raw [-1=open, +1=close] → clip[0,1] → invert(1-x)
            结果：训练数据 gripper 1=open / 0=close（OXE 标准约定）
          模型 decode 出来的 gripper ∈ [0, 1] 沿用此约定：
            > 0.5 → 模型在表达 "open"
            ≤ 0.5 → 模型在表达 "close"
          LIBERO env step 期望 raw 约定：-1=open, +1=close
          所以正确映射：g > threshold → -1（open），else +1（close）。
    """
    a = np.asarray(action, dtype=np.float32).copy()
    if args.action_clip:
        a[:6] = np.clip(a[:6], -1.0, 1.0)
    if args.invert_gripper:
        a[6] = -1.0 if a[6] > args.gripper_threshold else +1.0
    return a


# =================================================================
# 模型 / Env 构建
# =================================================================
def _load_model_and_processor(args: Args):
    """
    支持两种 ckpt 格式：
      1. **Full merged** ckpt（含 model.safetensors 7GB）→ AutoModel.from_pretrained 直接加载
      2. **PEFT/LoRA adapter** ckpt（含 adapter_config.json + adapter_model.safetensors）
         → 先加载 base_model（adapter_config.base_model_name_or_path 指定），
           再 PeftModel.attach adapter，最后 merge_and_unload 合并 LoRA。
         其中 adapter_config 的 modules_to_save (如 spatial_embed_tokens)
         会被 PEFT 完全 override 成 ckpt 版本，无需手动处理。
    """
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    ckpt_path = pathlib.Path(args.ckpt_path)
    is_peft = (ckpt_path / "adapter_config.json").exists() and not (ckpt_path / "model.safetensors").exists()

    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)

    if is_peft:
        adapter_cfg = json.load(open(ckpt_path / "adapter_config.json"))
        base_path = adapter_cfg["base_model_name_or_path"]
        logging.info(f"PEFT ckpt 格式：base={base_path}, adapter_dir={ckpt_path.name}")
        from peft import PeftModel  # 延迟 import；full merged ckpt 不必装 peft
        base_model = AutoModel.from_pretrained(
            base_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        # merge_and_unload 合并 LoRA 增量 → 返回与 full merged 同结构的纯 nn.Module
        model = peft_model.merge_and_unload().eval().cuda()
    else:
        model = (
            AutoModel.from_pretrained(
                args.ckpt_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation=args.attn_implementation,
            )
            .eval()
            .cuda()
        )

    chunk_size = int(processor.action_chunk_size)
    logging.info(f"模型加载完成: {args.ckpt_path}")
    logging.info(f"  dtype={args.dtype}, attn={args.attn_implementation}, action_chunk_size={chunk_size}, peft={is_peft}")
    logging.info(f"  unnorm_key='{args.unnorm_key}'，stats keys={list(processor.statistics.keys())[:5]}...")
    assert args.unnorm_key in processor.statistics, (
        f"unnorm_key='{args.unnorm_key}' 不在 processor.statistics 中。"
        f"可用 keys: {list(processor.statistics.keys())}"
    )

    # ============ 关键 sanity check：内参 + bin_policy 必须与训练对齐 ============
    # 训练时 scripts/intrinsics.json 把 libero_mix K 注入 processor，并用 scripts/gs_libero_mix.json
    # 重写 bin_policy；这两者会随 ckpt 一起 save_pretrained 落到 processor_config.json。
    # 推理时若任一缺失/错位，model 推出来的 spatial token 与 processor 的 bin 解码会错配，
    # 表现为 episode 全失败但又不抛异常——很难调。这里加一对显式断言把问题第一时间打出来。
    if args.unnorm_key in processor.dataset_intrinsics:
        K = processor.dataset_intrinsics[args.unnorm_key]
        logging.info(
            f"  📐 K[{args.unnorm_key}]: fx={K[0,0]:.3f} fy={K[1,1]:.3f} "
            f"cx={K[0,2]:.3f} cy={K[1,2]:.3f}"
        )
        # libero_mix 在 256 原始内参 309.02，processor 缩放到 image_processor.size(=224) 后应≈270.39
        if "libero" in args.unnorm_key:
            fx = float(K[0, 0])
            assert 260.0 < fx < 280.0, (
                f"LIBERO ckpt 但 processor K[0,0]={fx:.3f} 不在 [260,280] 区间，"
                f"可能 fallback 到了 bridge K(=218)。检查 ckpt processor_config.json 是否含 "
                f"intrinsic_config['{args.unnorm_key}']。"
            )
    else:
        logging.warning(
            f"⚠ processor.dataset_intrinsics 不含 '{args.unnorm_key}'，"
            f"会 fallback 到 default(=bridge) K！keys={list(processor.dataset_intrinsics.keys())}"
        )

    # bin_policy 校验：base ckpt 默认 r_bins[1]=0.5，gs_libero_mix 衍生 r_bins[1]≈0.3644。
    # 二者明显可分，可作为"是否成功用了 gs_libero_mix"的指示。
    bp = getattr(processor, "bin_policy", None)
    assert bp is not None and "translation" in bp, "processor.bin_policy 缺 translation 字段！"
    r_b1 = float(bp["translation"]["r_bins"][1])
    logging.info(f"  📊 bin_policy.translation.r_bins[1]={r_b1:.4f} (gs_libero_mix 衍生应≈0.3644)")
    assert 0.30 < r_b1 < 0.40, (
        f"bin_policy.translation.r_bins[1]={r_b1:.4f} 不在 gs_libero_mix 期望区间[0.30,0.40]，"
        f"可能误用了 base ckpt 的默认 bin。请检查训练时 spatial_embedding_adaption(gs_libero_mix.json) 是否生效。"
    )

    return model, processor, chunk_size


def _build_libero_env(task, resolution: int, seed: int):
    """初始化 LIBERO env + 返回 task 描述（与 openpi/main.py:_get_libero_env 等价）。"""
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


# =================================================================
# 推理一帧
# =================================================================
@torch.no_grad()
def _infer_action_chunk(model, processor, pil_image, prompt: str, unnorm_key: str) -> np.ndarray:
    """单帧推理：image+prompt → chunk_size × 7 的 action chunk（已 unnormalize）。"""
    inputs = processor(
        images=[pil_image],
        text=prompt,
        unnorm_key=unnorm_key,
        return_tensors="pt",
    ).to("cuda")
    generation_outputs = model.predict_action(inputs)
    actions = processor.decode_actions(generation_outputs, unnorm_key=unnorm_key)["actions"]
    # actions shape: [B=1, chunk_size, action_dim=7]
    return np.asarray(actions[0], dtype=np.float32)


# =================================================================
# 主 episode rollout
# =================================================================
def _run_episode(
    env, model, processor, chunk_size: int, ensembler, args: Args,
    initial_state, prompt: str, max_steps: int,
) -> tuple[bool, list]:
    env.reset()
    obs = env.set_init_state(initial_state)

    if ensembler is not None:
        ensembler.reset()

    action_plan: collections.deque = collections.deque()
    replay_images = []
    done = False
    t = 0

    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        primary = _preprocess_obs_image(obs["agentview_image"], args)
        replay_images.append(np.asarray(primary))

        if args.use_action_ensemble:
            # 每步都做一次 inference + ensemble（成本高但效果最稳）
            chunk = _infer_action_chunk(model, processor, primary, prompt, args.unnorm_key)
            action = ensembler.ensemble_action(chunk)
        else:
            # 不开 ensemble：用 chunk 队列重放，每 replan_steps 步重 inference
            if not action_plan:
                chunk = _infer_action_chunk(model, processor, primary, prompt, args.unnorm_key)
                # 取前 min(replan_steps, chunk_size) 个动作进队列
                steps_to_queue = min(args.replan_steps, chunk_size)
                action_plan.extend(chunk[:steps_to_queue])
            action = action_plan.popleft()

        action = _postprocess_action(action, args)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    return bool(done), replay_images


# =================================================================
# 主入口
# =================================================================
def eval_libero(args: Args) -> None:
    _set_seed_everywhere(args.seed, args.deterministic)
    logging.info(f"评测配置: {args}")

    model, processor, chunk_size = _load_model_and_processor(args)

    out_root = pathlib.Path(args.video_out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "eval_summary.json"

    benchmark_dict = benchmark.get_benchmark_dict()
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    overall_episodes, overall_successes = 0, 0
    suite_results = {}

    for suite_name in suites:
        assert suite_name in benchmark_dict, f"未知 suite: {suite_name}"
        max_steps = SUITE_MAX_STEPS[suite_name]
        task_suite = benchmark_dict[suite_name]()
        n_tasks = task_suite.n_tasks
        if args.max_tasks_per_suite > 0:
            n_tasks = min(n_tasks, args.max_tasks_per_suite)
        logging.info(f"=== Suite: {suite_name} | tasks={n_tasks} | max_steps={max_steps} ===")

        suite_episodes, suite_successes = 0, 0
        per_task_results = []

        suite_dir = out_root / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)

        for task_id in tqdm.tqdm(range(n_tasks), desc=f"{suite_name} tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_desc = _build_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            prompt = args.prompt_template.format(task=task_desc)

            task_episodes, task_successes = 0, 0

            ensembler = AdaptiveEnsembler(chunk_size, args.action_ensemble_alpha) \
                if args.use_action_ensemble else None

            for ep_idx in range(args.num_trials_per_task):
                env.seed(_episode_seed(args.seed, task_id, ep_idx))
                success, frames = _run_episode(
                    env, model, processor, chunk_size, ensembler, args,
                    initial_state=initial_states[ep_idx],
                    prompt=prompt,
                    max_steps=max_steps,
                )
                task_episodes += 1
                if success:
                    task_successes += 1

                if args.save_videos:
                    # 文件名包含：task_id / episode_id / 成功/失败 / 指令
                    # 指令做轻度清洗（空格→下划线，去掉文件系统不友好字符），过长截断到 80 字符
                    suffix = "成功" if success else "失败"
                    bad = ' /\\:*?"<>|\t\n\''
                    instr = "".join("_" if c in bad else c for c in task_desc).strip("_")[:80]
                    video_name = f"task{task_id:02d}_ep{ep_idx:03d}_{suffix}_{instr}.mp4"
                    imageio.mimwrite(
                        suite_dir / video_name,
                        frames,
                        fps=args.video_fps,
                    )

            sr = task_successes / max(task_episodes, 1)
            per_task_results.append({
                "task_id": task_id,
                "task_desc": task_desc,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": sr,
            })
            suite_episodes += task_episodes
            suite_successes += task_successes
            logging.info(f"  [{suite_name}] task {task_id}: SR={sr:.3f} ({task_successes}/{task_episodes}) | {task_desc}")

        suite_sr = suite_successes / max(suite_episodes, 1)
        logging.info(f"=== Suite {suite_name}: SR={suite_sr:.3f} ({suite_successes}/{suite_episodes}) ===")
        suite_results[suite_name] = {
            "episodes": suite_episodes,
            "successes": suite_successes,
            "success_rate": suite_sr,
            "per_task": per_task_results,
        }
        overall_episodes += suite_episodes
        overall_successes += suite_successes

    overall_sr = overall_successes / max(overall_episodes, 1)
    logging.info(f"=== Overall: SR={overall_sr:.3f} ({overall_successes}/{overall_episodes}) ===")

    summary = {
        "ckpt_path": args.ckpt_path,
        "unnorm_key": args.unnorm_key,
        "args": dataclasses.asdict(args),
        "overall": {
            "episodes": overall_episodes,
            "successes": overall_successes,
            "success_rate": overall_sr,
        },
        "per_suite": suite_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"已写 summary → {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # 直接 cli(Args) 避免 tyro 把字段嵌套到 --args.* 前缀；下划线自动转 dash 即可
    cli_args = tyro.cli(Args)
    eval_libero(cli_args)
