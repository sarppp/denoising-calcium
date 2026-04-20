"""Unified logging for training and inference.

Writes three synchronous streams:

1. **Console** — human-readable single-line key=value, printed with flush.
2. **Text log**  (``<out>/<name>.log``) — same as console, appended.
3. **JSONL**   (``<out>/<name>.jsonl``) — machine-readable rows, appended.

Every row carries a ``ts_sec`` (seconds since logger creation) and a
``kind`` field (``step``, ``epoch``, ``val``, ``infer``, ``bench``, …)
so downstream tools can filter without parsing the text form.

GPU peak memory is tracked automatically when ``cuda=True`` via
``torch.cuda.max_memory_allocated`` reset at each ``log(...)`` call; if
you want cumulative peak, don't reset between calls.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch


__all__ = ["RunLogger", "Timer", "format_bytes", "format_duration"]


def format_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TiB"


def format_duration(s: float) -> str:
    if s < 1:
        return f"{s * 1e3:.1f}ms"
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{int(h)}h{int(m):02d}m{s:04.1f}s"
    if m:
        return f"{int(m)}m{s:04.1f}s"
    return f"{s:.2f}s"


@dataclass
class Timer:
    """Minimal wall-clock timer with pretty ``__str__``."""

    label: str = ""
    t0: float = 0.0
    dt: float = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.dt = time.perf_counter() - self.t0

    def __str__(self) -> str:
        return f"{self.label}{'' if not self.label else ': '}{format_duration(self.dt)}"


class RunLogger:
    """Triple-sink logger: console + text + JSONL.

    Usage
    -----
        log = RunLogger(out_dir, name='train', cuda=True)
        log.log(kind='step', step=10, loss=0.42, lr=3e-4)
        log.close()

    Attributes
    ----------
    out_dir
        Parent directory that will receive ``<name>.log`` and ``<name>.jsonl``.
    """

    def __init__(
        self,
        out_dir: Path | str,
        name: str = "run",
        cuda: bool | None = None,
        echo: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.echo = bool(echo)
        self._jsonl = (self.out_dir / f"{name}.jsonl").open("a", buffering=1)
        self._txt = (self.out_dir / f"{name}.log").open("a", buffering=1)
        self.cuda = bool(torch.cuda.is_available()) if cuda is None else bool(cuda)
        self.t_start = time.time()
        self._t0 = time.perf_counter()

        # Header line identifying this run.
        self.log(
            kind="header",
            name=name,
            time=time.strftime("%Y-%m-%d %H:%M:%S"),
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            pid=os.getpid(),
        )

    # --------------------------------------------------------------------- #

    def log(self, **fields: Any) -> None:
        """Append a row to all three sinks.

        Any ``fields`` are free-form; conventionally include ``kind``.
        GPU memory is attached automatically as ``gpu_peak`` if ``self.cuda``.
        """
        row: dict[str, Any] = {"ts_sec": time.perf_counter() - self._t0}
        if self.cuda:
            try:
                row["gpu_peak_b"] = int(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        row.update(fields)

        # JSONL: exact fields.
        self._jsonl.write(json.dumps(row, default=str) + "\n")

        # Text: human-friendly.
        parts = [f"{k}={_fmt(v)}" for k, v in row.items() if k != "kind"]
        head = f"[{row.get('kind', '.')}]"
        line = f"{head} " + " ".join(parts)
        self._txt.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    # --------------------------------------------------------------------- #

    def close(self) -> None:
        self._jsonl.close()
        self._txt.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3e}"
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    if isinstance(v, int) and v.bit_length() > 20 and "gpu_peak_b" not in str(v):
        return format_bytes(v)
    return str(v)
