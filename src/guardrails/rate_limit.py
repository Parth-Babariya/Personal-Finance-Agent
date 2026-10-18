"""In-process sliding-window rate limiter.

State lives in memory, so limits apply per API process. For several replicas,
swap this for a shared store (e.g. Redis) behind the same interface.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        return hits

    def hit(self, key: str) -> float | None:
        """Record a hit. Returns ``None`` if allowed, else seconds until the next slot frees up."""
        if self.limit <= 0:
            return None
        with self._lock:
            now = self._clock()
            hits = self._prune(key, now)
            if len(hits) >= self.limit:
                return max(0.0, hits[0] + self.window - now)
            hits.append(now)
            return None

    def retry_after(self, key: str) -> float | None:
        """Like ``hit`` but without recording: ``None`` if under the limit."""
        if self.limit <= 0:
            return None
        with self._lock:
            now = self._clock()
            hits = self._prune(key, now)
            return max(0.0, hits[0] + self.window - now) if len(hits) >= self.limit else None

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)
