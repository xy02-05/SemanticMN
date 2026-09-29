#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def directory_snapshot(path: Path) -> tuple[int, int, int]:
    file_count = 0
    total_bytes = 0
    latest_mtime_ns = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        stat = item.stat()
        file_count += 1
        total_bytes += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return file_count, total_bytes, latest_mtime_ns


def checkpoint_step(path: Path) -> int:
    return int(path.name.removeprefix("checkpoint-"))


def checkpoint_complete(path: Path) -> bool:
    state_path = path / "training_state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state.get("checkpoint_step") == checkpoint_step(path)


def wait_until_stable(path: Path, polls: int, interval: float) -> tuple[int, int, int]:
    previous = None
    stable_count = 0
    while path.is_dir() and not path.is_symlink():
        current = directory_snapshot(path)
        if current == previous:
            stable_count += 1
        else:
            stable_count = 0
            previous = current
        if stable_count >= polls:
            return current
        time.sleep(interval)
    raise RuntimeError(f"checkpoint disappeared while waiting for stability: {path}")


def run_rsync(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--partial",
            f"{source}/",
            f"{destination}/",
        ],
        check=True,
    )


def archive_checkpoint(
    checkpoint: Path,
    archive_root: Path,
    stable_polls: int,
    stable_interval: float,
) -> None:
    step = checkpoint_step(checkpoint)
    final_archive = archive_root / checkpoint.name
    if final_archive.is_dir():
        source_snapshot = directory_snapshot(checkpoint)
        archive_snapshot = directory_snapshot(final_archive)
        if source_snapshot[:2] != archive_snapshot[:2]:
            raise RuntimeError(
                f"existing archive differs for step {step}: "
                f"source={source_snapshot[:2]} archive={archive_snapshot[:2]}"
            )
    else:
        source_snapshot = wait_until_stable(checkpoint, stable_polls, stable_interval)
        staging = archive_root / (
            f".{checkpoint.name}.incoming.{socket.gethostname()}.{os.getpid()}"
        )
        if staging.exists():
            shutil.rmtree(staging)
        run_rsync(checkpoint, staging)
        archive_snapshot = directory_snapshot(staging)
        if source_snapshot[:2] != archive_snapshot[:2]:
            raise RuntimeError(
                f"archive verification failed for step {step}: "
                f"source={source_snapshot[:2]} archive={archive_snapshot[:2]}"
            )
        staging.rename(final_archive)

    local_backup = checkpoint.with_name(f".{checkpoint.name}.local")
    checkpoint.rename(local_backup)
    checkpoint.symlink_to(final_archive, target_is_directory=True)
    if not (checkpoint / "training_state.json").is_file():
        checkpoint.unlink()
        local_backup.rename(checkpoint)
        raise RuntimeError(f"archive symlink verification failed for step {step}")
    shutil.rmtree(local_backup)
    print(
        f"[archive] step={step} local={checkpoint} archive={final_archive} "
        f"files={archive_snapshot[0]} bytes={archive_snapshot[1]}",
        flush=True,
    )


def recover_interrupted_links(local_root: Path, archive_root: Path) -> None:
    for backup in sorted(local_root.glob(".checkpoint-*.local")):
        checkpoint = local_root / backup.name[1:].removesuffix(".local")
        archive = archive_root / checkpoint.name
        if checkpoint.is_symlink() and (checkpoint / "training_state.json").is_file():
            shutil.rmtree(backup)
        elif archive.is_dir() and (archive / "training_state.json").is_file():
            if checkpoint.exists() or checkpoint.is_symlink():
                checkpoint.unlink()
            checkpoint.symlink_to(archive, target_is_directory=True)
            shutil.rmtree(backup)
        else:
            if checkpoint.exists() or checkpoint.is_symlink():
                checkpoint.unlink()
            backup.rename(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将本地完整 checkpoint 异步归档到共享盘并回建软链接"
    )
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument("--stable-interval", type=float, default=15.0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument(
        "--allowed-steps",
        type=int,
        nargs="+",
        help="只归档指定 step；不传时保持原有的全部归档行为",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_steps = set(args.allowed_steps) if args.allowed_steps else None
    args.local_root.mkdir(parents=True, exist_ok=True)
    args.archive_root.mkdir(parents=True, exist_ok=True)
    recover_interrupted_links(args.local_root, args.archive_root)

    print(
        f"[archive] watching local={args.local_root} archive={args.archive_root}",
        flush=True,
    )
    while True:
        if args.stop_file is not None and args.stop_file.exists():
            print(f"[archive] stop file detected: {args.stop_file}", flush=True)
            return
        checkpoints = sorted(
            (
                path
                for path in args.local_root.glob("checkpoint-*")
                if path.is_dir() and not path.is_symlink()
                and (allowed_steps is None or checkpoint_step(path) in allowed_steps)
            ),
            key=checkpoint_step,
        )
        for checkpoint in checkpoints:
            if checkpoint_complete(checkpoint):
                archive_checkpoint(
                    checkpoint,
                    args.archive_root,
                    args.stable_polls,
                    args.stable_interval,
                )
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
