"""Record process-local PyTorch CUDA peaks for external trainer commands."""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import threading
from pathlib import Path

import torch

_OUTPUT = os.environ.get("TRAINING_COMPARISON_TORCH_MEMORY")
if _OUTPUT and torch.cuda.is_available():
    _output_path = Path(_OUTPUT)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    _stop = threading.Event()
    _peak_allocated = 0
    _peak_reserved = 0

    def _sample() -> None:
        global _peak_allocated, _peak_reserved
        _peak_allocated = max(_peak_allocated, torch.cuda.max_memory_allocated())
        _peak_reserved = max(_peak_reserved, torch.cuda.max_memory_reserved())

    def _poll() -> None:
        while not _stop.wait(0.05):
            _sample()

    _poll_thread = threading.Thread(target=_poll, daemon=True)
    _poll_thread.start()

    @atexit.register
    def _write_peak() -> None:
        _stop.set()
        _poll_thread.join()
        _sample()
        output = _output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        lock_path = output.with_suffix(output.suffix + ".lock")
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
            value = {
                "device": torch.cuda.get_device_name(),
                "max_memory_allocated_bytes": max(
                    _peak_allocated, previous.get("max_memory_allocated_bytes", 0)
                ),
                "max_memory_reserved_bytes": max(
                    _peak_reserved, previous.get("max_memory_reserved_bytes", 0)
                ),
            }
            output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
