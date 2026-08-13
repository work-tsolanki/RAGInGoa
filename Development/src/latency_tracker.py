import time
import json
from functools import wraps
from typing import Dict, List
from config import DEBUG

class LatencyTracker:
    """Track latency of each component."""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}

    def record(self, component: str, latency_ms: float):
        """Record a latency measurement."""
        if component not in self.measurements:
            self.measurements[component] = []

        self.measurements[component].append(latency_ms)

        if DEBUG:
            print(f"  [{component}] {latency_ms:.1f}ms")

    def get_stats(self) -> Dict:
        """Get P50, P70, P100 for each component."""
        stats = {}
        for component, latencies in self.measurements.items():
            latencies = sorted(latencies)
            n = len(latencies)

            stats[component] = {
                "p50": latencies[int(n * 0.5)],
                "p70": latencies[int(n * 0.7)] if n > 1 else latencies[0],
                "p100": latencies[-1],
                "mean": sum(latencies) / n,
                "count": n
            }

        return stats

    def print_summary(self):
        """Print latency summary."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("LATENCY SUMMARY")
        print("="*60)

        for component, data in stats.items():
            print(f"\n{component}:")
            print(f"  P50: {data['p50']:.1f}ms")
            print(f"  P70: {data['p70']:.1f}ms")
            print(f"  P100: {data['p100']:.1f}ms")
            print(f"  Mean: {data['mean']:.1f}ms")
            print(f"  Count: {data['count']}")


# Global tracker instance
_tracker = LatencyTracker()

def track_latency(component: str):
    """Decorator to track function latency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            _tracker.record(component, latency_ms)
            return result
        return wrapper
    return decorator

def get_tracker():
    """Get global tracker instance."""
    return _tracker
