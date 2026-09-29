#!/usr/bin/env python3

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


ROOT = Path("/mnt/bn/2d-videos/xy/work/mirror_neuron")
LOG = ROOT / "logs/gr00t_bridge_h20_4gpu_aligned_local.log"
MIGRATION_LOG = ROOT / "logs/gr00t_bridge_h20_4gpu_aligned_migration.log"
STATUS_FILE = ROOT / "logs/gr00t_bridge_h20_4gpu_monitor_status.json"
MONITOR_LOG = ROOT / "logs/gr00t_bridge_h20_4gpu_monitor.log"
SOURCE = Path("/tmp/gr00t_bridge_h20_4gpu_aligned/gr00t_bridge_h20_4gpu_aligned")
ARCHIVE = ROOT / "runs/gr00t_bridge_h20_4gpu_aligned"
TRAIN_SESSION = "gr00t_bridge_h20_4gpu_aligned"
MIGRATOR_SESSION = "gr00t_bridge_checkpoint_migrator"
TARGET_STEP = 3000
POLL_SECONDS = 30
STALL_SECONDS = 15 * 60

STEP_PATTERN = re.compile(rb"(\d+)/20000")
LOSS_PATTERN = re.compile(rb"'loss':\s*([-+0-9.eE]+)")
FATAL_PATTERNS = (
    b"ChildFailedError",
    b"ProcessGroupNCCL.cpp:632",
    b"CUDA out of memory",
    b"OutOfMemoryError",
    b"Traceback (most recent call last)",
    b"Training completed!",
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


def process_commands() -> list[tuple[int, str]]:
    commands = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        commands.append((int(proc_dir.name), command.strip()))
    return commands


def memory_guard_quota(parts: list[str]) -> str | None:
    if (
        len(parts) < 2
        or not Path(parts[0]).name.startswith("python")
        or parts[1] != "/mnt/bn/2d-videos/xy/memory_guard.py"
    ):
        return None
    try:
        return parts[parts.index("--quota-gib") + 1]
    except (ValueError, IndexError):
        return ""


def terminate_bad_resource_processes(commands: list[tuple[int, str]]) -> list[str]:
    terminated = []
    for pid, command in commands:
        parts = command.split()
        is_gpu_wave = bool(parts) and parts[0] == (
            "/mnt/bn/2d-videos/xy/tools/gpus/gpu_wave_load"
        )
        quota = memory_guard_quota(parts)
        is_wrong_guard = quota is not None and quota != "850"
        if not (is_gpu_wave or is_wrong_guard):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append(f"{pid}:{command}")
        except ProcessLookupError:
            pass

    for session in ("gpu-utilization", "memory-watchdog"):
        if tmux_session_exists(session):
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            terminated.append(f"tmux:{session}")
    return terminated


def read_log_state() -> tuple[int, float | None, bytes]:
    if not LOG.is_file():
        return 0, None, b""
    data = LOG.read_bytes()
    step_matches = STEP_PATTERN.findall(data)
    loss_matches = LOSS_PATTERN.findall(data)
    step = max((int(value) for value in step_matches), default=0)
    loss = float(loss_matches[-1]) if loss_matches else None
    return step, loss, data


def write_status(status: dict) -> None:
    temporary = STATUS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True))
    os.replace(temporary, STATUS_FILE)


def checkpoint_2500_state() -> str:
    source = SOURCE / "checkpoint-2500"
    archive = ARCHIVE / "checkpoint-2500"
    if source.is_symlink() and archive.joinpath("trainer_state.json").is_file():
        return "migrated"
    if source.joinpath("trainer_state.json").is_file():
        return "complete_local"
    if source.exists():
        return "saving"
    return "absent"


def fail(message: str, status: dict, exit_code: int) -> None:
    status["health"] = "failed"
    status["error"] = message
    status["updated_at"] = time.strftime("%F %T")
    write_status(status)
    append_log(f"ERROR: {message}")
    raise SystemExit(exit_code)


def main() -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    initial_size = LOG.stat().st_size if LOG.exists() else 0
    last_step = 0
    last_progress_at = time.monotonic()
    append_log(f"monitoring training through step {TARGET_STEP}")

    while not stop_requested:
        commands = process_commands()
        terminated = terminate_bad_resource_processes(commands)
        if terminated:
            append_log(f"removed conflicting resource process(es): {terminated}")
            commands = process_commands()

        step, loss, log_data = read_log_state()
        if step > last_step:
            last_step = step
            last_progress_at = time.monotonic()

        train_processes = [
            pid
            for pid, command in commands
            if "launch_finetune.py" in command
            and "--experiment_name gr00t_bridge_h20_4gpu_aligned" in command
        ]
        guards = []
        for pid, command in commands:
            parts = command.split()
            if memory_guard_quota(parts) == "850":
                guards.append(pid)
        checkpoint_state = checkpoint_2500_state()
        status = {
            "health": "running",
            "step": step,
            "loss": loss,
            "train_process_count": len(train_processes),
            "guard_process_count": len(guards),
            "checkpoint_2500": checkpoint_state,
            "train_session": tmux_session_exists(TRAIN_SESSION),
            "migrator_session": tmux_session_exists(MIGRATOR_SESSION),
            "updated_at": time.strftime("%F %T"),
        }
        write_status(status)

        if not status["train_session"] or not train_processes:
            fail("training process or tmux session exited", status, 20)
        if not status["migrator_session"]:
            fail("checkpoint migrator exited", status, 21)
        if len(guards) != 1:
            fail(f"expected one 850 GiB guard, found {len(guards)}", status, 22)
        if loss is not None and not math.isfinite(loss):
            fail(f"non-finite loss detected: {loss}", status, 23)

        new_log = log_data[initial_size:]
        for pattern in FATAL_PATTERNS:
            if pattern in new_log:
                fail(f"fatal log marker detected: {pattern.decode()}", status, 24)

        if time.monotonic() - last_progress_at > STALL_SECONDS:
            fail(
                f"training made no progress for {STALL_SECONDS} seconds at step {step}",
                status,
                25,
            )

        if step >= TARGET_STEP:
            if checkpoint_state != "migrated":
                append_log(
                    f"step {step} reached; waiting for checkpoint-2500 migration"
                )
            else:
                status["health"] = "target_reached"
                status["updated_at"] = time.strftime("%F %T")
                write_status(status)
                append_log(
                    f"target reached at step {step}; checkpoint-2500 migrated"
                )
                return

        time.sleep(POLL_SECONDS)

    append_log("monitor stopped by signal")


if __name__ == "__main__":
    main()
