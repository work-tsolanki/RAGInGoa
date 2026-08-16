import sys

sys.path.insert(0, '.')

from src.rate_limiter import RateLimiter


def test_allows_up_to_max_requests():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False


def test_different_keys_have_independent_budgets():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False
    assert limiter.allow("5.6.7.8") is True


def test_sweep_evicts_keys_with_no_hits_left_in_window():
    """Regression test for unbounded memory growth: a distinct IP that
    makes one request and never returns must not keep a permanent dict
    entry forever - see RateLimiter._sweep."""
    import time

    limiter = RateLimiter(max_requests=100, window_seconds=0.05)
    for i in range(250):
        limiter.allow(f"10.0.0.{i}")
    assert len(limiter._hits) == 250

    time.sleep(0.1)  # let the window expire for all of them
    limiter._sweep(time.time())

    assert len(limiter._hits) == 0


def test_sweep_keeps_keys_with_hits_still_in_window():
    import time

    limiter = RateLimiter(max_requests=100, window_seconds=60)
    limiter.allow("1.2.3.4")
    limiter._sweep(time.time())
    assert "1.2.3.4" in limiter._hits


def test_allow_triggers_periodic_sweep_automatically():
    """End-to-end: allow() itself eventually sweeps without an explicit
    _sweep() call, once enough calls have accumulated."""
    import time

    limiter = RateLimiter(max_requests=100, window_seconds=0.05)
    for i in range(RateLimiter._SWEEP_EVERY_N_CALLS - 1):
        limiter.allow(f"10.0.0.{i}")
    assert len(limiter._hits) == RateLimiter._SWEEP_EVERY_N_CALLS - 1

    time.sleep(0.1)
    limiter.allow("final-call-triggers-sweep")  # crosses the threshold

    # The sweep ran as part of this call, evicting everything expired
    # before this one's own (fresh) hit was recorded.
    assert len(limiter._hits) == 1
    assert "final-call-triggers-sweep" in limiter._hits


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
