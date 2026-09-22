"""Run one trainer while sampling its process-tree GPU memory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _descendants(root: int) -> set[int]:
    found = {root}
    pending = [root]
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = (int(value) for value in children_path.read_text().split())
        except (FileNotFoundError, PermissionError):
            continue
        for child in children:
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _gpu_memory_mib(pids: set[int]) -> int:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    total = 0
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in pids:
            try:
                total += int(fields[1])
            except ValueError:
                pass
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak_mib = 0
    samples = 0
    with args.log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            peak_mib = max(peak_mib, _gpu_memory_mib(_descendants(process.pid)))
            samples += 1
            time.sleep(0.1)
        returncode = process.wait()
    result = {
        "command": command,
        "cwd": str(args.cwd.resolve()),
        "nvidia_smi_peak_mib": peak_mib,
        "returncode": returncode,
        "runtime_seconds": time.monotonic() - started,
        "samples": samples,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
