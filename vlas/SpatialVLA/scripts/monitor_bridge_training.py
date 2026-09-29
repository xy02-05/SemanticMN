#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


STEP_PATTERN = re.compile(r"Step\s+(\d+)/")
LOSS_PATTERN = re.compile(
    r"总损失:([0-9.eE+-]+).*?VLA_avg:([0-9.eE+-]+).*?Align_avg:([0-9.eE+-]+)"
)
DSN_PATTERN = re.compile(
    r"DSN diff=([0-9.eE+-]+|N/A) recon=([0-9.eE+-]+|N/A)"
)
ERROR_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|ChildFailedError|"
    r"DistBackendError|NCCL.*(?:error|timeout)|\bnan\b|\binf\b",
    re.IGNORECASE,
)


def read_new_text(path: Path, offset: int) -> tuple[str, int]:
    if not path.is_file():
        return "", offset
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("r", encoding="utf-8", errors="replace") as file:
        file.seek(offset)
        text = file.read()
        return text, file.tell()


def process_alive(pid_path: Path) -> bool:
    if not pid_path.is_file():
        return False
    pid = int(pid_path.read_text().strip())
    return Path(f"/proc/{pid}").exists()


def gpu_summary() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return " | ".join(line.strip() for line in result.stdout.splitlines())


def checkpoint_status(run_root: Path, archive_root: Path, step: int) -> tuple[bool, str]:
    local_checkpoint = run_root / f"checkpoint-{step}"
    archive_checkpoint = archive_root / f"checkpoint-{step}"
    local_ok = (
        local_checkpoint.is_symlink()
        and (local_checkpoint / "training_state.json").is_file()
        and local_checkpoint.resolve() == archive_checkpoint.resolve()
    )
    archive_ok = (archive_checkpoint / "training_state.json").is_file()
    return local_ok and archive_ok, (
        f"local_symlink={local_checkpoint.is_symlink()} "
        f"archive_state={archive_ok}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="持续监控两组 SpatialVLA 训练直到 checkpoint-1000 归档并跨过 step 1500"
    )
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--archive-root", type=Path, action="append", required=True)
    parser.add_argument("--target-step", type=int, default=1500)
    parser.add_argument("--checkpoint-step", type=int, default=1000)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--status-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.run_root) != len(args.archive_root):
        raise ValueError("--run-root and --archive-root counts must match")

    state = {
        str(run_root): {
            "offset": 0,
            "step": 0,
            "total_loss": None,
            "vla_loss": None,
            "alignment_loss": None,
            "dsn_diff_loss": None,
            "dsn_total_loss": None,
            "checkpoint_ok": False,
            "started_at": time.monotonic(),
        }
        for run_root in args.run_root
    }
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    while True:
        all_done = True
        rows = []
        for run_root, archive_root in zip(
            args.run_root, args.archive_root, strict=True
        ):
            run_state = state[str(run_root)]
            text, run_state["offset"] = read_new_text(
                run_root / "training.log", run_state["offset"]
            )
            error = ERROR_PATTERN.search(text)
            if error:
                raise RuntimeError(
                    f"training error in {run_root}: {error.group(0)}\n{text[-4000:]}"
                )
            for match in STEP_PATTERN.finditer(text):
                run_state["step"] = max(run_state["step"], int(match.group(1)))
            for match in LOSS_PATTERN.finditer(text):
                run_state["total_loss"] = float(match.group(1))
                run_state["vla_loss"] = float(match.group(2))
                run_state["alignment_loss"] = float(match.group(3))
            for match in DSN_PATTERN.finditer(text):
                if match.group(1) != "N/A":
                    run_state["dsn_diff_loss"] = float(match.group(1))
                if match.group(2) != "N/A":
                    run_state["dsn_total_loss"] = float(match.group(2))

            checkpoint_ok, checkpoint_detail = checkpoint_status(
                run_root, archive_root, args.checkpoint_step
            )
            run_state["checkpoint_ok"] = checkpoint_ok
            alive = process_alive(run_root / "train.pid")
            done = run_state["step"] >= args.target_step and checkpoint_ok
            startup_grace_elapsed = time.monotonic() - run_state["started_at"]
            if not done and not alive and startup_grace_elapsed > 900:
                raise RuntimeError(
                    f"training process exited before target in {run_root}: "
                    f"step={run_state['step']} checkpoint={checkpoint_detail}"
                )
            all_done = all_done and done
            rows.append(
                {
                    "run": run_root.name,
                    "step": run_state["step"],
                    "total_loss": run_state["total_loss"],
                    "vla_loss": run_state["vla_loss"],
                    "alignment_loss": run_state["alignment_loss"],
                    "dsn_diff_loss": run_state["dsn_diff_loss"],
                    "dsn_total_loss": run_state["dsn_total_loss"],
                    "process_alive": alive,
                    "checkpoint_1000_ok": checkpoint_ok,
                    "checkpoint_detail": checkpoint_detail,
                }
            )

        status = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": gpu_summary(),
            "runs": rows,
            "complete": all_done,
        }
        args.status_file.write_text(
            json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if all_done:
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
