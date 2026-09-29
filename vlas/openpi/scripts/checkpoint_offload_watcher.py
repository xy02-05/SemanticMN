#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import time


@dataclass(frozen=True)
class TreeState:
    files: int
    bytes: int
    newest_mtime_ns: int


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def scan_tree(path: Path) -> TreeState:
    files = 0
    total_bytes = 0
    newest_mtime_ns = 0
    for root, _, names in os.walk(path):
        for name in names:
            file_path = Path(root) / name
            stat = file_path.stat()
            files += 1
            total_bytes += stat.st_size
            newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
    return TreeState(files, total_bytes, newest_mtime_ns)


def checkpoint_complete(path: Path) -> bool:
    return (
        (path / "model.safetensors").is_file()
        and (path / "optimizer.pt").is_file()
        and (path / "metadata.pt").is_file()
    )


def has_open_files(path: Path) -> bool:
    prefix = f"{path.resolve()}{os.sep}"
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            descriptors = list((process_dir / "fd").iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target == str(path.resolve()) or target.startswith(prefix):
                return True
    return False


def copy_tree(source: Path, partial: Path) -> None:
    partial.mkdir(parents=True, exist_ok=True)
    source_files: set[Path] = set()
    for root, directories, names in os.walk(source):
        relative_root = Path(root).relative_to(source)
        target_root = partial / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (target_root / directory).mkdir(parents=True, exist_ok=True)
        for name in names:
            source_file = Path(root) / name
            relative_file = relative_root / name
            source_files.add(relative_file)
            target_file = partial / relative_file
            source_stat = source_file.stat()
            if target_file.exists():
                target_stat = target_file.stat()
                if (
                    target_stat.st_size == source_stat.st_size
                    and target_stat.st_mtime_ns == source_stat.st_mtime_ns
                ):
                    continue
            shutil.copy2(source_file, target_file)

    for root, _, names in os.walk(partial):
        relative_root = Path(root).relative_to(partial)
        for name in names:
            target_file = Path(root) / name
            if relative_root / name not in source_files:
                target_file.unlink()


def replace_with_symlink(source: Path, target: Path) -> None:
    backup = source.with_name(f".{source.name}.local-backup")
    if backup.exists():
        raise FileExistsError(backup)
    source.rename(backup)
    try:
        source.symlink_to(target, target_is_directory=True)
    except Exception:
        backup.rename(source)
        raise
    shutil.rmtree(backup)


def offload(source: Path, archive_root: Path) -> None:
    target = archive_root / source.name
    partial = archive_root / f"{source.name}.partial"
    source_state = scan_tree(source)

    if target.exists():
        if scan_tree(target) != source_state:
            raise ValueError(f"existing archive differs from source: {target}")
    else:
        copy_tree(source, partial)
        partial_state = scan_tree(partial)
        if partial_state.files != source_state.files or partial_state.bytes != source_state.bytes:
            raise ValueError(
                f"copy verification failed: source={source_state}, partial={partial_state}"
            )
        partial.rename(target)

    replace_with_symlink(source, target)
    log(f"OFFLOADED {source.name}: files={source_state.files} bytes={source_state.bytes}")


def iter_checkpoints(local_root: Path):
    checkpoints = []
    for path in local_root.iterdir():
        if path.is_dir() and not path.is_symlink() and path.name.isdigit():
            checkpoints.append((int(path.name), path))
    for _, path in sorted(checkpoints):
        yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive stable OpenPI checkpoints and link them back.")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--min-free-gib", type=float, default=200.0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stable_checks < 2:
        raise ValueError("stable-checks must be at least 2")
    args.local_root.mkdir(parents=True, exist_ok=True)
    args.archive_root.mkdir(parents=True, exist_ok=True)
    previous: dict[Path, TreeState] = {}
    stable_counts: dict[Path, int] = {}
    once_iterations = 0
    log(f"watching local={args.local_root} archive={args.archive_root}")

    while True:
        free_gib = shutil.disk_usage(args.local_root).free / 1024**3
        if free_gib < args.min_free_gib:
            log(f"WARNING local free space {free_gib:.1f} GiB < {args.min_free_gib:.1f} GiB")

        for checkpoint in iter_checkpoints(args.local_root):
            if not checkpoint_complete(checkpoint) or has_open_files(checkpoint):
                previous.pop(checkpoint, None)
                stable_counts.pop(checkpoint, None)
                continue
            state = scan_tree(checkpoint)
            if previous.get(checkpoint) == state:
                stable_counts[checkpoint] = stable_counts.get(checkpoint, 1) + 1
            else:
                previous[checkpoint] = state
                stable_counts[checkpoint] = 1
            if stable_counts[checkpoint] >= args.stable_checks:
                offload(checkpoint, args.archive_root)
                previous.pop(checkpoint, None)
                stable_counts.pop(checkpoint, None)

        once_iterations += 1
        if args.once and once_iterations >= args.stable_checks:
            return
        if args.stop_file is not None and args.stop_file.exists():
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
