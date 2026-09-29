#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import re
import shutil
import signal
import time


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
stop_requested = False


def request_stop(signum, frame):
    del signum, frame
    global stop_requested
    stop_requested = True


def directory_signature(path: Path) -> tuple[int, int, int]:
    file_count = 0
    total_size = 0
    latest_mtime_ns = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            stat = entry.stat()
            file_count += 1
            total_size += stat.st_size
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return file_count, total_size, latest_mtime_ns


def has_temporary_files(path: Path) -> bool:
    return any(entry.name.startswith(".tmp") for entry in path.rglob("*"))


def has_open_files(path: Path) -> bool:
    prefix = f"{path.resolve()}{os.sep}"
    for proc_dir in Path("/proc").glob("[0-9]*"):
        fd_dir = proc_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if target == str(path) or target.startswith(prefix):
                return True
    return False


def verify_copy(source: Path, destination: Path) -> None:
    source_files = {
        entry.relative_to(source): entry.stat().st_size
        for entry in source.rglob("*")
        if entry.is_file()
    }
    destination_files = {
        entry.relative_to(destination): entry.stat().st_size
        for entry in destination.rglob("*")
        if entry.is_file()
    }
    if source_files != destination_files:
        raise RuntimeError(f"Checkpoint copy verification failed: {source}")


def copy_checkpoint(source: Path, archive_root: Path) -> None:
    destination = archive_root / source.name
    if destination.exists():
        raise FileExistsError(f"Archive checkpoint already exists: {destination}")

    temporary_destination = archive_root / f".{source.name}.copying-{os.getpid()}"
    local_backup = source.parent / f".{source.name}.local-{os.getpid()}"
    temporary_link = source.parent / f".{source.name}.link-{os.getpid()}"

    if temporary_destination.exists():
        shutil.rmtree(temporary_destination)
    shutil.copytree(source, temporary_destination, copy_function=shutil.copy2)
    verify_copy(source, temporary_destination)
    os.rename(temporary_destination, destination)

    # 本地路径只在所有文件完成复制和校验后才替换为指向 mnt 的软链接。
    temporary_link.symlink_to(destination)
    os.rename(source, local_backup)
    os.replace(temporary_link, source)
    shutil.rmtree(local_backup)
    print(f"[{time.strftime('%F %T')}] migrated {source.name} -> {destination}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate completed checkpoints to mnt.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--stable-seconds", type=float, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source.mkdir(parents=True, exist_ok=True)
    args.archive.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    observed: dict[Path, tuple[tuple[int, int, int], float]] = {}
    print(
        f"[{time.strftime('%F %T')}] watching {args.source} -> {args.archive}",
        flush=True,
    )

    while not stop_requested:
        now = time.monotonic()
        candidates = sorted(
            (
                entry
                for entry in args.source.iterdir()
                if entry.is_dir()
                and not entry.is_symlink()
                and CHECKPOINT_PATTERN.fullmatch(entry.name)
            ),
            key=lambda entry: int(CHECKPOINT_PATTERN.fullmatch(entry.name).group(1)),
        )

        for checkpoint in candidates:
            if not (checkpoint / "trainer_state.json").is_file():
                observed.pop(checkpoint, None)
                continue
            if has_temporary_files(checkpoint):
                observed.pop(checkpoint, None)
                continue

            signature = directory_signature(checkpoint)
            previous = observed.get(checkpoint)
            if previous is None or previous[0] != signature:
                observed[checkpoint] = (signature, now)
                continue
            if now - previous[1] < args.stable_seconds or has_open_files(checkpoint):
                continue

            copy_checkpoint(checkpoint, args.archive)
            observed.pop(checkpoint, None)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
