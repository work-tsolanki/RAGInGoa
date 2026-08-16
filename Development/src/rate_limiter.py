import time
from collections import defaultdict, deque


class RateLimiter:
    """In-memory sliding-window rate limiter, keyed by an arbitrary string
    (IP address). Single-process/single-machine only (see fly.toml -
    min_machines_running=1, no horizontal scaling) - a multi-machine
    deployment would need a shared store instead."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True
