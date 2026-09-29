#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shutil
import time


@dataclass(frozen=True)
class TreeState:
    files: int
    bytes: int
    latest_mtime_ns: int


def scan_tree(path: Path) -> TreeState:
    files = 0
    total_bytes = 0
    latest_mtime_ns = 0
    for root, _, names in os.walk(path):
        for name in names:
            file_path = Path(root) / name
            if file_path.is_symlink():
                continue
            stat = file_path.stat()
            files += 1
            total_bytes += stat.st_size
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return TreeState(files, total_bytes, latest_mtime_ns)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%F %T')}] {message}", flush=True)


def checkpoint_is_complete(path: Path) -> bool:
    required_any = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    if not any((path / name).exists() for name in required_any):
        return False
    return (path / "trainer_state.json").exists()


def has_temporary_files(path: Path) -> bool:
    return any(entry.name.startswith(".tmp") for entry in path.rglob("*"))


def has_open_files(path: Path) -> bool:
    path_prefix = f"{path.resolve()}{os.sep}"
    for process_dir in Path("/proc").glob("[0-9]*"):
        file_descriptors = process_dir / "fd"
        try:
            descriptors = list(file_descriptors.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target == str(path) or target.startswith(path_prefix):
                return True
    return False


def copy_tree(source: Path, partial: Path) -> None:
    partial.mkdir(parents=True, exist_ok=True)
    source_files = set()
    for root, directories, names in os.walk(source):
        relative_root = Path(root).relative_to(source)
        target_root = partial / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (target_root / directory).mkdir(parents=True, exist_ok=True)
        for name in names:
            source_file = Path(root) / name
            relative_file = relative_root / name
            target_file = partial / relative_file
            source_files.add(relative_file)
            source_stat = source_file.stat()
            if target_file.exists():
                target_stat = target_file.stat()
                if (
                    target_stat.st_size == source_stat.st_size
                    and target_stat.st_mtime_ns == source_stat.st_mtime_ns
                ):
                    continue
            shutil.copy2(source_file, target_file)

    # 清理中断前遗留但源目录已经不存在的文件，保证最终校验严格一致。
    for root, _, names in os.walk(partial):
        relative_root = Path(root).relative_to(partial)
        for name in names:
            target_file = Path(root) / name
            if relative_root / name not in source_files:
                target_file.unlink()


def replace_source_with_symlink(source: Path, target: Path) -> None:
    backup = source.with_name(f".{source.name}.local-backup")
    if backup.exists():
        raise FileExistsError(f"stale local backup exists: {backup}")
    source.rename(backup)
    try:
        source.symlink_to(target, target_is_directory=True)
    except Exception:
        backup.rename(source)
        raise
    shutil.rmtree(backup)


def offload_final_model(local_root: Path, archive_root: Path) -> None:
    completed_step = (local_root / ".training_complete").read_text().strip()
    if not completed_step.isdigit():
        raise ValueError(".training_complete must contain the completed global step")
    final_root = archive_root / f"final_model_step-{completed_step}"
    final_root.mkdir(parents=True, exist_ok=True)
    patterns = (
        "model*.safetensors",
        "pytorch_model*.bin",
        "model*.index.json",
        "pytorch_model*.index.json",
        "training_args.bin",
        "config.json",
    )
    files = sorted({path for pattern in patterns for path in local_root.glob(pattern)})
    for source in files:
        if source.is_symlink():
            continue
        partial = final_root / f"{source.name}.partial"
        target = final_root / source.name
        source_stat = source.stat()
        if not target.exists():
            if not partial.exists() or partial.stat().st_size != source_stat.st_size:
                shutil.copy2(source, partial)
            if partial.stat().st_size != source_stat.st_size:
                raise ValueError(f"final model copy verification failed: {source}")
            partial.replace(target)
        if target.stat().st_size != source_stat.st_size:
            raise ValueError(f"existing final model archive differs: {target}")
        backup = source.with_name(f".{source.name}.local-backup")
        source.rename(backup)
        try:
            source.symlink_to(target)
        except Exception:
            backup.rename(source)
            raise
        backup.unlink()
        log(f"offloaded final model file {source.name}: bytes={source_stat.st_size}")


def offload_checkpoint(source: Path, archive_root: Path) -> None:
    target = archive_root / source.name
    partial = archive_root / f"{source.name}.partial"

    if target.exists():
        source_state = scan_tree(source)
        target_state = scan_tree(target)
        if source_state.files != target_state.files or source_state.bytes != target_state.bytes:
            raise ValueError(f"existing archive differs from local source: {target}")
    else:
        copy_tree(source, partial)
        source_state = scan_tree(source)
        partial_state = scan_tree(partial)
        if source_state.files != partial_state.files or source_state.bytes != partial_state.bytes:
            raise ValueError(
                f"copy verification failed for {source}: source={source_state}, partial={partial_state}"
            )
        partial.rename(target)

    replace_source_with_symlink(source, target)
    log(f"offloaded {source.name}: files={source_state.files} bytes={source_state.bytes}")


def iter_checkpoints(local_root: Path):
    checkpoints = []
    for path in local_root.glob("checkpoint-*"):
        if path.is_dir() and not path.is_symlink():
            suffix = path.name.removeprefix("checkpoint-")
            if suffix.isdigit():
                checkpoints.append((int(suffix), path))
    for _, path in sorted(checkpoints):
        yield path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move completed local GR00T checkpoints to /mnt and link them back."
    )
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--min-free-gib", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stable_checks < 2:
        raise ValueError("stable-checks must be at least 2")
    args.local_root.mkdir(parents=True, exist_ok=True)
    args.archive_root.mkdir(parents=True, exist_ok=True)

    previous_states: dict[Path, TreeState] = {}
    stable_counts: dict[Path, int] = {}
    log(
        f"watching local={args.local_root} archive={args.archive_root} "
        f"stable_checks={args.stable_checks}"
    )

    while True:
        disk = shutil.disk_usage(args.local_root)
        free_gib = disk.free / 1024**3
        if free_gib < args.min_free_gib:
            log(
                f"WARNING local free space is {free_gib:.1f} GiB, "
                f"below {args.min_free_gib:.1f} GiB"
            )

        active_paths = set()
        for checkpoint in iter_checkpoints(args.local_root):
            active_paths.add(checkpoint)
            if not checkpoint_is_complete(checkpoint):
                continue
            if has_temporary_files(checkpoint) or has_open_files(checkpoint):
                previous_states.pop(checkpoint, None)
                stable_counts.pop(checkpoint, None)
                continue
            state = scan_tree(checkpoint)
            if previous_states.get(checkpoint) == state:
                stable_counts[checkpoint] = stable_counts.get(checkpoint, 1) + 1
            else:
                stable_counts[checkpoint] = 1
                previous_states[checkpoint] = state

            if stable_counts[checkpoint] >= args.stable_checks:
                offload_checkpoint(checkpoint, args.archive_root)
                previous_states.pop(checkpoint, None)
                stable_counts.pop(checkpoint, None)

        if (args.local_root / ".training_complete").exists():
            offload_final_model(args.local_root, args.archive_root)

        for stale_path in previous_states.keys() - active_paths:
            previous_states.pop(stale_path, None)
            stable_counts.pop(stale_path, None)

        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
