"""Замер вычислительной стоимости ядра на самой длинной записи.

Каждое входное сообщение проходит через тот же код, что вызывает узел ROS.
Измеряются время обработки сообщения, пиковая память Python и то, во сколько
раз обработка быстрее реального времени. Время DDS и публикации не входит,
его измеряет latency_probe на стенде ROS.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'odometria'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bagdata import load  # noqa: E402
from odometria.core import OdometryCore  # noqa: E402
from odometria.track import TrackMap  # noqa: E402
from odometria.traction import TractionModel  # noqa: E402
from replay import stream  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = load(args.bag, args.cache)
    events = stream(data)
    core = OdometryCore(track_map=TrackMap.load(), model=TractionModel.load())
    times = {'wheel': [], 'cmd': [], 'fix': []}
    outputs = 0
    begin_all = time.perf_counter()
    for _, kind, name, stamp, value in events:
        begin = time.perf_counter()
        if kind == 0:
            outputs += len(core.on_wheel(name, stamp, value))
            key = 'wheel'
        elif kind == 1:
            outputs += len(core.on_command(stamp, value))
            key = 'cmd'
        else:
            core.on_fix(name, stamp, *value)
            key = 'fix'
        times[key].append((time.perf_counter() - begin) * 1e3)
    wall = time.perf_counter() - begin_all
    # второй проход только ради пиковой памяти: tracemalloc замедляет код
    tracemalloc.start()
    probe = OdometryCore(track_map=TrackMap.load(), model=TractionModel.load())
    for _, kind, name, stamp, value in events:
        if kind == 0:
            probe.on_wheel(name, stamp, value)
        elif kind == 1:
            probe.on_command(stamp, value)
        else:
            probe.on_fix(name, stamp, *value)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    duration = float(data['cmd'][-1, 0] - data['cmd'][0, 0])

    def pct(v):
        v = np.asarray(v)
        return {'n': int(len(v)), 'p50_ms': round(float(np.percentile(v, 50)), 4),
                'p99_ms': round(float(np.percentile(v, 99)), 4), 'max_ms': round(float(v.max()), 3)}

    report = {
        'bag': args.bag.name, 'duration_s': round(duration, 1), 'outputs': outputs,
        'output_rate_hz': round(outputs / duration, 2),
        'processing': {k: pct(v) for k, v in times.items() if v},
        'wall_s': round(wall, 2), 'realtime_factor': round(duration / wall, 1),
        'cpu_share_of_one_core': round(wall / duration, 4),
        'python_peak_alloc_mb': round(peak / 2 ** 20, 2),
        'note': 'чистое ядро на одном ядре CPU; время DDS измеряет latency_probe',
        'platform': sys.platform,
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
