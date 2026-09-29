import dataclasses
import importlib
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import torch
import yaml


OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[2]
STANDARD_LIBERO_ROOT = OPENPI_ROOT / "third_party" / "libero"
DEFAULT_LIBERO_PRO_ROOT = OPENPI_ROOT.parent / "LIBERO-PRO"
LIBERO_PRO_SUFFIXES = ("temp", "lan", "object", "swap", "task", "env")
SINGLE_PERTURBATION_SUFFIXES = ("lan", "object", "swap", "task", "env")
BASE_SUITE_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
SUFFIX_TO_FLAG = {
    "lan": "use_language",
    "object": "use_object",
    "swap": "use_swap",
    "task": "use_task",
    "env": "use_environment",
}


@dataclasses.dataclass(frozen=True)
class LiberoRuntime:
    """运行时导出的 LIBERO 模块句柄。"""

    benchmark: Any
    get_libero_path: Any
    offscreen_env_cls: Any
    package_root: pathlib.Path | None
    benchmark_root: pathlib.Path | None


def get_base_suite_name(task_suite_name: str) -> str:
    for suffix in LIBERO_PRO_SUFFIXES:
        suffix_token = f"_{suffix}"
        if task_suite_name.endswith(suffix_token):
            candidate = task_suite_name[: -len(suffix_token)]
            if candidate in BASE_SUITE_NAMES:
                return candidate
    return task_suite_name


def get_suite_suffix(task_suite_name: str) -> str | None:
    base_suite_name = get_base_suite_name(task_suite_name)
    if base_suite_name == task_suite_name:
        return None
    return task_suite_name[len(base_suite_name) + 1 :]


def is_libero_pro_suite(task_suite_name: str) -> bool:
    return get_suite_suffix(task_suite_name) is not None


def _resolve_package_root(task_suite_name: str) -> pathlib.Path | None:
    env_root = os.environ.get("LIBERO_PACKAGE_ROOT")
    if env_root:
        return pathlib.Path(env_root).resolve()

    if is_libero_pro_suite(task_suite_name) and DEFAULT_LIBERO_PRO_ROOT.exists():
        return DEFAULT_LIBERO_PRO_ROOT.resolve()

    if STANDARD_LIBERO_ROOT.exists():
        return STANDARD_LIBERO_ROOT.resolve()

    return None


def _resolve_benchmark_root(package_root: pathlib.Path | None) -> pathlib.Path | None:
    if package_root is None:
        return None
    benchmark_root = package_root / "libero" / "libero"
    if benchmark_root.exists():
        return benchmark_root
    return None


def _default_config_dir(package_root: pathlib.Path | None) -> pathlib.Path:
    if package_root is None:
        return pathlib.Path("/tmp/openpi_libero_runtime/default")
    slug = package_root.name.lower().replace("-", "_")
    return pathlib.Path("/tmp/openpi_libero_runtime") / slug


def _prepare_libero_config(package_root: pathlib.Path | None) -> pathlib.Path | None:
    benchmark_root = _resolve_benchmark_root(package_root)
    if benchmark_root is None:
        return None

    config_dir = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", _default_config_dir(package_root)))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(package_root / "libero" / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    # 原子写入：先写临时文件再 rename，避免多进程并发时读到被截断的空文件
    import tempfile
    config_yaml = config_dir / "config.yaml"
    fd, tmp_path = tempfile.mkstemp(dir=str(config_dir), suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    os.replace(tmp_path, str(config_yaml))
    return benchmark_root


def _ensure_import_path(package_root: pathlib.Path | None) -> None:
    if package_root is None:
        return
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)
    # pip editable install 会让 sys.modules 缓存旧版 libero，
    # 必须清除后重新导入才能用到 LIBERO-PRO 注册的物体（如 bigger_akita_black_bowl）
    stale = [k for k in sys.modules if k == "libero" or k.startswith("libero.")]
    for k in stale:
        del sys.modules[k]


def _load_libero_pro_eval_config(package_root: pathlib.Path) -> dict[str, Any]:
    def resolve_repo_path(raw_path: str) -> pathlib.Path:
        path = pathlib.Path(raw_path)
        if path.is_absolute():
            return path

        candidates = []
        prefix = f"./{package_root.name}/"
        if raw_path.startswith(prefix):
            candidates.append((package_root / raw_path[len(prefix) :]).resolve())
        candidates.append((package_root / raw_path).resolve())
        candidates.append((package_root.parent / raw_path).resolve())

        prefix = f"{package_root.name}/"
        if raw_path.startswith(prefix):
            candidates.append((package_root.parent / raw_path).resolve())

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    config_path = pathlib.Path(
        os.environ.get("LIBERO_PRO_EVAL_CONFIG", package_root / "evaluation_config.yaml")
    )
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    ood_task_configs = config.get("ood_task_configs", {}) or {}
    resolved_ood_task_configs = {}
    for key, value in ood_task_configs.items():
        resolved_ood_task_configs[key] = str(resolve_repo_path(value))
    config["ood_task_configs"] = resolved_ood_task_configs

    script_path = config.get("script_path", "./notebooks/generate_init_states.py")
    config["script_path"] = str(resolve_repo_path(script_path))
    return config


def _count_suite_files(directory: pathlib.Path, suffix: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(f"*{suffix}")))


def _count_init_states(init_file: pathlib.Path) -> int:
    states = torch.load(init_file, map_location="cpu", weights_only=False)
    return len(states)


def _suite_has_enough_inits(init_dir: pathlib.Path, num_trials_per_task: int) -> bool:
    init_files = sorted(init_dir.glob("*.pruned_init"))
    if not init_files:
        return False
    return _count_init_states(init_files[0]) >= num_trials_per_task


def _copy_init_suite(base_init_dir: pathlib.Path, target_init_dir: pathlib.Path) -> None:
    target_init_dir.mkdir(parents=True, exist_ok=True)
    for init_file in base_init_dir.glob("*.pruned_init"):
        shutil.copy2(init_file, target_init_dir / init_file.name)


def _inits_match_base(base_init_dir: pathlib.Path, target_init_dir: pathlib.Path) -> bool:
    """检查 target init files 是否与 base 字节一致。"""
    import filecmp

    if not target_init_dir.exists():
        return False
    base_files = sorted(base_init_dir.glob("*.pruned_init"))
    for bf in base_files:
        tf = target_init_dir / bf.name
        if not tf.exists() or not filecmp.cmp(bf, tf, shallow=False):
            return False
    return True


def _generate_init_suite(
    package_root: pathlib.Path,
    eval_config: dict[str, Any],
    target_bddl_dir: pathlib.Path,
    target_init_dir: pathlib.Path,
    num_trials_per_task: int,
) -> None:
    target_init_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    package_root_str = str(package_root)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{package_root_str}:{old_pythonpath}" if old_pythonpath else package_root_str
    )
    subprocess.run(
        [
            sys.executable,
            eval_config["script_path"],
            "--bddl_base_dir",
            str(target_bddl_dir),
            "--output_dir",
            str(target_init_dir),
            "--num_inits",
            str(num_trials_per_task),
        ],
        check=True,
        env=env,
    )


def _build_pro_suite_bddl(
    package_root: pathlib.Path,
    benchmark_root: pathlib.Path,
    task_suite_name: str,
    seed: int,
) -> pathlib.Path:
    base_suite_name = get_base_suite_name(task_suite_name)
    suffix = get_suite_suffix(task_suite_name)
    if suffix not in SINGLE_PERTURBATION_SUFFIXES:
        raise ValueError(
            f"当前只支持单扰动 suite 的按需生成，缺失目录时不自动生成: {task_suite_name}"
        )

    target_bddl_dir = benchmark_root / "bddl_files" / task_suite_name
    base_bddl_dir = benchmark_root / "bddl_files" / base_suite_name
    target_bddl_dir.mkdir(parents=True, exist_ok=True)

    eval_config = _load_libero_pro_eval_config(package_root)
    perturbation = importlib.import_module("perturbation")
    flags = perturbation.PerturbFlags(**{SUFFIX_TO_FLAG[suffix]: True})
    pipeline = perturbation.BDDLCombinedPerturbator(configs=eval_config["ood_task_configs"])

    for bddl_file in sorted(base_bddl_dir.glob("*.bddl")):
        with bddl_file.open("r", encoding="utf-8") as f:
            content = f.read()
        new_content = pipeline.perturb_content(
            content=content,
            task_suite_name=base_suite_name,
            task_name=bddl_file.stem,
            flags=flags,
            seed=seed,
        )
        with (target_bddl_dir / bddl_file.name).open("w", encoding="utf-8") as f:
            f.write(new_content)

    return target_bddl_dir


def _ensure_libero_pro_suite_assets(
    package_root: pathlib.Path,
    benchmark_root: pathlib.Path,
    task_suite_name: str,
    num_trials_per_task: int,
    seed: int,
) -> None:
    if not is_libero_pro_suite(task_suite_name):
        return

    target_bddl_dir = benchmark_root / "bddl_files" / task_suite_name
    target_init_dir = benchmark_root / "init_files" / task_suite_name
    base_suite_name = get_base_suite_name(task_suite_name)
    suffix = get_suite_suffix(task_suite_name)
    base_bddl_dir = benchmark_root / "bddl_files" / base_suite_name
    base_init_dir = benchmark_root / "init_files" / base_suite_name

    # 优先使用官方预生成的 bddl + init（从 HuggingFace 下载）。
    # 只有当文件缺失时才 fallback 到自动生成。
    bddl_ok = _count_suite_files(target_bddl_dir, ".bddl") == _count_suite_files(base_bddl_dir, ".bddl")
    init_ok = target_init_dir.exists() and _suite_has_enough_inits(target_init_dir, num_trials_per_task)

    if bddl_ok and init_ok:
        return

    logging.info("Preparing LIBERO-PRO suite assets (missing): %s  bddl_ok=%s init_ok=%s", task_suite_name, bddl_ok, init_ok)

    if not bddl_ok:
        _build_pro_suite_bddl(package_root, benchmark_root, task_suite_name, seed)

    if not init_ok:
        eval_config = _load_libero_pro_eval_config(package_root)
        _generate_init_suite(
            package_root=package_root,
            eval_config=eval_config,
            target_bddl_dir=target_bddl_dir,
            target_init_dir=target_init_dir,
            num_trials_per_task=num_trials_per_task,
        )


def prepare_libero_runtime(
    task_suite_name: str,
    num_trials_per_task: int,
    seed: int,
) -> LiberoRuntime:
    """
    在导入 libero 之前准备路径和 config。

    这里把逻辑收口在一处，避免把评测主流程改得很散。
    """

    # 兼容新版本 PyTorch 对 .pruned_init 的默认 weights_only 限制。
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    package_root = _resolve_package_root(task_suite_name)
    benchmark_root = _prepare_libero_config(package_root)
    _ensure_import_path(package_root)

    if package_root is not None and benchmark_root is not None and is_libero_pro_suite(task_suite_name):
        _ensure_libero_pro_suite_assets(
            package_root=package_root,
            benchmark_root=benchmark_root,
            task_suite_name=task_suite_name,
            num_trials_per_task=num_trials_per_task,
            seed=seed,
        )

    libero_root = importlib.import_module("libero.libero")
    benchmark = importlib.import_module("libero.libero.benchmark")
    envs = importlib.import_module("libero.libero.envs")
    return LiberoRuntime(
        benchmark=benchmark,
        get_libero_path=libero_root.get_libero_path,
        offscreen_env_cls=envs.OffScreenRenderEnv,
        package_root=package_root,
        benchmark_root=benchmark_root,
    )
