"""Measure the pure observer step cost; ROS transport is measured separately."""
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'odometria'))
from odometria.estimator import Observer  # noqa: E402


def sample(observer, index):
    t = index * 0.02
    u = 8 if index % 1200 < 600 else -4
    speed = max(0.0, 8.0 + 1.2 * ((index // 50) % 10))
    wheel = speed * 3.6
    return observer.step(t, u, wheel, wheel)


def main():
    observer = Observer()
    timings = []
    for index in range(50001):
        begin = perf_counter_ns()
        sample(observer, index)
        if index:
            timings.append((perf_counter_ns() - begin) / 1e6)
    tracemalloc.start()
    memory_observer = Observer()
    for index in range(10001):
        sample(memory_observer, index)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings.sort()
    result = {
        'samples': len(timings),
        'platform': sys.platform,
        'latency_ms_p50': statistics.median(timings),
        'latency_ms_p95': timings[int(0.95 * len(timings))],
        'latency_ms_p99': timings[int(0.99 * len(timings))],
        'latency_ms_max': timings[-1],
        'python_peak_alloc_bytes_10000_steps': peak,
        'scope': 'pure Python observer only; excludes ROS communication and scheduling',
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
