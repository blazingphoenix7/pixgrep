from __future__ import annotations

import time


def measure(embed_fn, images: list, warmup: int = 1) -> float:
    """Images-per-second for embed_fn, ignoring the first `warmup` images."""
    if not images:
        return 0.0
    if warmup > 0:
        embed_fn(images[:warmup])
    timed = images[warmup:] or images
    t0 = time.perf_counter()
    embed_fn(timed)
    dt = time.perf_counter() - t0
    return len(timed) / dt if dt > 0 else 0.0
