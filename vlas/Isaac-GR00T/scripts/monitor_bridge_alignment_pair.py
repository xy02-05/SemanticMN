#!/usr/bin/env python3

import ast
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


ROOT = Path("/mnt/bn/2d-videos/xy/work/mirror_neuron")
LOG_DIR = ROOT / "logs"
STATUS_FILE = LOG_DIR / "gr00t_bridge_alignment_pair_3000_status.json"
MONITOR_LOG = LOG_DIR / "gr00t_bridge_alignment_pair_3000_monitor.log"
TARGET_STEP = 3000
CHECKPOINT_STEP = 2500
POLL_SECONDS = 30
STALL_SECONDS = 30 * 60
MAX_STEPS = 20000
WARMUP_END_STEP = 2000
LOGGING_STEPS = 10

RUNS = {
    "qwen": {
        "log": LOG_DIR / "gr00t_bridge_mirror_neuron_qwen_w1_final_20k.log",
        "local": Path(
            "/tmp/mirror_neuron/gr00t_runs/qwen_w1_final_20k/"
            "mirror_neuron_qwen_w1_final_20k"
        ),
        "archive": ROOT
        / "runs/gr00t_bridge_mirror_neuron_archive/qwen_w1_final_20k",
        "train_session": "gr00t-train-qwen-w1-final",
        "watchdog_session": "gr00t-ckpt-qwen_w1_final_20k",
        "master_port": "29761",
    },
    "egohod": {
        "log": LOG_DIR / "gr00t_bridge_mirror_neuron_egohod_w1_final_20k.log",
        "local": Path(
            "/tmp/mirror_neuron/gr00t_runs/egohod_w1_final_20k/"
            "mirror_neuron_egohod_w1_final_20k"
        ),
        "archive": ROOT
        / "runs/gr00t_bridge_mirror_neuron_archive/egohod_w1_final_20k",
        "train_session": "gr00t-train-egohod-w1-final",
        "watchdog_session": "gr00t-ckpt-egohod_w1_final_20k",
        "master_port": "29762",
    },
}

STEP_PATTERN = re.compile(rb"(\d+)/20000")
METRIC_PATTERN = re.compile(rb"\{'loss': [^\r\n]+\}")
FATAL_PATTERNS = (
    b"ChildFailedError",
    b"CUDA out of memory",
    b"OutOfMemoryError",
    b"Traceback (most recent call last)",
    b"ProcessGroupNCCL.cpp",
    b"received death signal",
    b"Signal 9 (SIGKILL)",
)
stop_requested = False


def request_stop(signum, frame):
    del signum, frame
    global stop_requested
    stop_requested = True


def append_log(message: str) -> None:
    line = f"[{time.strftime('%F %T')}] {message}"
    with MONITOR_LOG.open("a") as file:
        file.write(f"{line}\n")
    print(line, flush=True)


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def process_table() -> list[dict]:
    processes = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (
                (proc_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            stat = (proc_dir / "stat").read_text()
            stat_tail = stat[stat.rfind(")") + 2 :].split()
            processes.append(
                {
                    "pid": int(proc_dir.name),
                    "ppid": int(stat_tail[1]),
                    "command": command,
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return processes


def read_log_state(path: Path) -> tuple[int, dict, bytes]:
    if not path.is_file():
        return 0, {}, b""
    data = path.read_bytes()
    step = max((int(value) for value in STEP_PATTERN.findall(data)), default=0)
    metric_matches = METRIC_PATTERN.findall(data)
    metric = {}
    if metric_matches:
        try:
            metric = ast.literal_eval(metric_matches[-1].decode(errors="replace"))
        except (SyntaxError, ValueError):
            metric = {}
    return step, metric, data


def checkpoint_state(run: dict) -> str:
    local = run["local"] / f"checkpoint-{CHECKPOINT_STEP}"
    archive = run["archive"] / f"checkpoint-{CHECKPOINT_STEP}"
    if local.is_symlink() and archive.joinpath("trainer_state.json").is_file():
        return "migrated"
    if local.joinpath("trainer_state.json").is_file():
        return "complete_local"
    if local.exists():
        return "saving"
    return "absent"


def gpu_state() -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        index, memory_used, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "memory_used_mib": int(memory_used),
                "utilization_gpu_pct": int(utilization),
            }
        )
    return rows


def write_status(status: dict) -> None:
    temporary = STATUS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True))
    os.replace(temporary, STATUS_FILE)


def fail(message: str, status: dict, exit_code: int) -> None:
    status["health"] = "failed"
    status["error"] = message
    status["updated_at"] = time.strftime("%F %T")
    write_status(status)
    append_log(f"ERROR: {message}")
    raise SystemExit(exit_code)


def validate_metric(name: str, step: int, metric: dict) -> str | None:
    if not metric:
        return None
    for key in (
        "loss",
        "action_loss",
        "alignment_loss",
        "weighted_alignment_loss",
        "grad_norm",
    ):
        value = metric.get(key)
        if value is None or not math.isfinite(float(value)):
            return f"{name}: non-finite or missing {key}: {value}"

    detached = metric.get("alignment_backbone_detached")
    # 前 10% 只截断 alignment loss 到 backbone 的梯度；action loss 始终存在。
    if step < WARMUP_END_STEP and detached != 1.0:
        return f"{name}: alignment backbone detached too early at step {step}: {detached}"
    # 进度条可能先于对应的 logging step 落盘，因此给最新指标一个日志周期的缓冲。
    if step >= WARMUP_END_STEP + LOGGING_STEPS and detached != 0.0:
        return f"{name}: alignment backbone still detached at step {step}: {detached}"
    return None


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    initial_sizes = {
        name: run["log"].stat().st_size if run["log"].exists() else 0
        for name, run in RUNS.items()
    }
    last_steps = {name: 0 for name in RUNS}
    last_progress_at = {name: time.monotonic() for name in RUNS}
    last_reported_bucket = {name: -1 for name in RUNS}
    append_log(
        f"monitoring qwen and egohod through step {TARGET_STEP}; "
        f"checkpoint-{CHECKPOINT_STEP} must be migrated"
    )

    while not stop_requested:
        processes = process_table()
        run_statuses = {}
        for name, run in RUNS.items():
            step, metric, log_data = read_log_state(run["log"])
            if step > last_steps[name]:
                last_steps[name] = step
                last_progress_at[name] = time.monotonic()

            torchruns = [
                process
                for process in processes
                if "torchrun" in process["command"]
                and f"--master_port={run['master_port']}" in process["command"]
            ]
            rank_count = 0
            if len(torchruns) == 1:
                rank_count = sum(
                    process["ppid"] == torchruns[0]["pid"]
                    and "launch_finetune.py" in process["command"]
                    for process in processes
                )
            watchdogs = [
                process
                for process in processes
                if "checkpoint_offload_watchdog.py" in process["command"]
                and f"--archive-root {run['archive']}" in process["command"]
            ]
            run_statuses[name] = {
                "step": step,
                "metric": metric,
                "checkpoint_2500": checkpoint_state(run),
                "train_session": tmux_session_exists(run["train_session"]),
                "watchdog_session": tmux_session_exists(run["watchdog_session"]),
                "torchrun_count": len(torchruns),
                "rank_count": rank_count,
                "watchdog_process_count": len(watchdogs),
            }

            if not run_statuses[name]["train_session"] or len(torchruns) != 1:
                fail(f"{name}: training tmux or torchrun exited", {"runs": run_statuses}, 20)
            if rank_count != 4:
                fail(f"{name}: expected 4 direct training ranks, found {rank_count}", {"runs": run_statuses}, 21)
            if (
                not run_statuses[name]["watchdog_session"]
                or len(watchdogs) != 1
            ):
                fail(f"{name}: checkpoint watchdog exited", {"runs": run_statuses}, 22)

            metric_error = validate_metric(name, step, metric)
            if metric_error is not None:
                fail(metric_error, {"runs": run_statuses}, 23)
            for pattern in FATAL_PATTERNS:
                if pattern in log_data[initial_sizes[name] :]:
                    fail(
                        f"{name}: fatal log marker: {pattern.decode()}",
                        {"runs": run_statuses},
                        24,
                    )
            if time.monotonic() - last_progress_at[name] > STALL_SECONDS:
                fail(
                    f"{name}: no progress for {STALL_SECONDS} seconds at step {step}",
                    {"runs": run_statuses},
                    25,
                )

            bucket = step // 100
            if bucket > last_reported_bucket[name]:
                last_reported_bucket[name] = bucket
                append_log(
                    f"{name}: step={step}, loss={metric.get('loss')}, "
                    f"action_loss={metric.get('action_loss')}, "
                    f"alignment_loss={metric.get('alignment_loss')}, "
                    f"detached={metric.get('alignment_backbone_detached')}, "
                    f"checkpoint={run_statuses[name]['checkpoint_2500']}"
                )

        status = {
            "health": "running",
            "target_step": TARGET_STEP,
            "max_steps": MAX_STEPS,
            "warmup_end_step": WARMUP_END_STEP,
            "runs": run_statuses,
            "gpus": gpu_state(),
            "updated_at": time.strftime("%F %T"),
        }
        write_status(status)

        target_reached = all(
            run_status["step"] >= TARGET_STEP for run_status in run_statuses.values()
        )
        checkpoints_migrated = all(
            run_status["checkpoint_2500"] == "migrated"
            for run_status in run_statuses.values()
        )
        if target_reached and checkpoints_migrated:
            status["health"] = "target_reached"
            status["updated_at"] = time.strftime("%F %T")
            write_status(status)
            append_log(
                "both runs reached step 3000 with finite metrics, correct warmup "
                "state, and migrated checkpoint-2500"
            )
            return

        time.sleep(POLL_SECONDS)

    append_log("monitor stopped by signal")


if __name__ == "__main__":
    main()
