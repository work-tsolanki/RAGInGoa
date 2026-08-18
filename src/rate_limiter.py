import time
from collections import defaultdict, deque


class RateLimiter:
    """In-memory sliding-window rate limiter, keyed by an arbitrary string
    (IP address). Single-process/single-machine only (see fly.toml -
    min_machines_running=1, no horizontal scaling) - a multi-machine
    deployment would need a shared store instead.

    Periodically sweeps out keys with no hits left in the current window,
    rather than letting every distinct IP that's ever made a request keep a
    permanent dict entry - on a public, unauthenticated endpoint, an
    attacker cycling through many source IPs (one or two requests each,
    then switching) would otherwise grow this dict without bound for as
    long as the process stays up (this app is meant to stay up through the
    whole judging window - see fly.toml's auto_stop_machines=false)."""

    _SWEEP_EVERY_N_CALLS = 500

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits = defaultdict(deque)
        self._calls_since_sweep = 0

    def allow(self, key: str) -> bool:
        now = time.time()
        hits = self._hits[key]
        self._trim(hits, now)

        if len(hits) >= self.max_requests:
            allowed = False
        else:
            hits.append(now)
            allowed = True

        # Runs after this key's own hit is recorded (when allowed), so the
        # sweep can never delete the entry this exact call just touched -
        # doing this before recording the hit could race with a brand-new
        # key's still-empty deque and silently drop it.
        self._calls_since_sweep += 1
        if self._calls_since_sweep >= self._SWEEP_EVERY_N_CALLS:
            self._sweep(now)
            self._calls_since_sweep = 0

        return allowed

    def _trim(self, hits: deque, now: float) -> None:
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()

    def _sweep(self, now: float) -> None:
        for key in list(self._hits.keys()):
            hits = self._hits[key]
            self._trim(hits, now)
            if not hits:
                del self._hits[key]
