"""Held-out 2D route test using GNSS only for initialization and scoring."""
import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path
import sqlite3
import sys

from analyze_dataset import decode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'odometria'))
from odometria.estimator import Observer  # noqa: E402
from odometria.route import Projector, load_routes  # noqa: E402


def evaluate(bag):
    db = next(bag.glob('*.db3'))
    connection = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    topics = {i: (name, kind) for i, name, kind in connection.execute(
        'SELECT id, name, type FROM topics')}
    events = []
    fixes = {'master': [], 'rover': []}
    for received_ns, topic_id, blob in connection.execute(
            'SELECT timestamp, topic_id, data FROM messages ORDER BY timestamp'):
        name, kind = topics[topic_id]
        t = received_ns * 1e-9
        if name.endswith('/fix') and '/sensing/gnss/' in name:
            fixes['master' if '/master/' in name else 'rover'].append(
                (t, decode(blob, kind)[1]))
        elif name.startswith('/vehicle/'):
            events.append((t, name, decode(blob, kind)[1]))
    connection.close()
    source = 'master' if fixes['master'] else 'rover'
    reference = fixes[source]
    if not reference:
        return {'bag': bag.name, 'error': 'no GNSS fix'}
    lat, lon, alt = reference[0][1]
    heading = None
    if fixes['master'] and fixes['rover']:
        mlat, mlon, _ = fixes['master'][0][1]
        rlat, rlon, _ = fixes['rover'][0][1]
        heading = (6371000 * math.radians(rlon - mlon) * math.cos(math.radians(mlat)),
                   6371000 * math.radians(rlat - mlat))
    projector = Projector(load_routes())
    projector.initialize(lat, lon, alt, *(heading or (None, None)))
    observer = Observer()
    latest = {'front': None, 'rear': None, 'command': None}
    outputs = []
    for t, name, value in events:
        if name.endswith('driver_position_cmd'):
            latest['command'] = (t, value)
            continue
        side = 'front' if 'front_' in name else 'rear'
        latest[side] = (t, value)
        def current(key, timeout):
            sample = latest[key]
            if sample is None:
                return None, timeout + 1.0
            return sample[1], t - sample[0]
        command, command_age = current('command', observer.cfg.command_timeout)
        front, front_age = current('front', observer.cfg.velocity_timeout)
        rear, rear_age = current('rear', observer.cfg.velocity_timeout)
        result = observer.step(t, command, front, rear, command_age=command_age,
                               front_age=front_age, rear_age=rear_age)
        outputs.append((t, projector.position(result.distance), result.distance))
    times = [row[0] for row in outputs]
    errors_map, errors_baseline = [], []
    errors_map_3d, errors_baseline_3d = [], []
    last = None
    origin = projector.route.metric(lat, lon, alt)
    for t, fix in reference:
        index = bisect_left(times, t)
        index = min((i for i in (index - 1, index) if 0 <= i < len(times)),
                    key=lambda i: abs(times[i] - t), default=None)
        if index is None or abs(times[index] - t) > 0.05:
            continue
        ref = projector.route.metric(*fix)
        actual = tuple(a - b for a, b in zip(ref, origin))
        _, predicted, distance = outputs[index]
        map_error = math.dist(actual[:2], predicted[:2])
        baseline_error = math.dist(actual[:2], (distance, 0.0))
        errors_map.append(map_error ** 2)
        errors_baseline.append(baseline_error ** 2)
        errors_map_3d.append(math.dist(actual, predicted) ** 2)
        errors_baseline_3d.append(math.dist(actual, (distance, 0.0, 0.0)) ** 2)
        last = (map_error, baseline_error)
    return {'bag': bag.name, 'source': source, 'map_start_error_m': projector.map_error,
            'map_source': projector.route.source,
            'map_used': projector.start_s is not None, 'direction': projector.direction,
            'pairs': len(errors_map),
            'map_rmse_2d_m': math.sqrt(sum(errors_map) / len(errors_map)) if errors_map else None,
            'baseline_rmse_2d_m': math.sqrt(sum(errors_baseline) / len(errors_baseline)) if errors_baseline else None,
            'map_rmse_3d_m': math.sqrt(sum(errors_map_3d) / len(errors_map_3d)) if errors_map_3d else None,
            'baseline_rmse_3d_m': math.sqrt(sum(errors_baseline_3d) / len(errors_baseline_3d)) if errors_baseline_3d else None,
            'final_errors_m': last}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path, nargs='*')
    parser.add_argument('--data', type=Path, help='evaluate all bag folders here')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    bags = args.bag + (sorted(p.parent for p in args.data.glob('*/metadata.yaml'))
                       if args.data else [])
    if not bags:
        parser.error('pass bag folders or --data')
    report = json.dumps([evaluate(bag) for bag in bags], indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    else:
        print(report)


if __name__ == '__main__':
    main()
