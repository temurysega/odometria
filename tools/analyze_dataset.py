"""Inspect supplied rosbag2 SQLite recordings without a ROS installation.

GNSS is decoded only in this offline scoring tool. The deployed node imports
neither this file nor any GNSS topic.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'odometria'))
from odometria.estimator import Observer  # noqa: E402


def unpack_header(blob):
    if blob[:4] != b'\x00\x01\x00\x00':
        raise ValueError('expected little endian ROS 2 CDR')
    sec, nsec, length = struct.unpack_from('<iII', blob, 4)
    end = 16 + length
    if length < 1 or end > len(blob) or blob[end - 1] != 0:
        raise ValueError('invalid CDR header string')
    return sec + nsec * 1e-9, end


def aligned(offset, size):
    return 4 + ((offset - 4 + size - 1) // size) * size


def decode(blob, kind):
    stamp, offset = unpack_header(blob)
    if kind == 'tram_vehicle_msgs/msg/VelocitySensor':
        value = struct.unpack_from('<d', blob, aligned(offset, 8))[0]
    elif kind == 'tram_vehicle_msgs/msg/DriverControllerCommand':
        value = struct.unpack_from('<b', blob, offset)[0]
    elif kind == 'geometry_msgs/msg/TwistStamped':
        value = struct.unpack_from('<ddd', blob, aligned(offset, 8))
    else:
        raise ValueError(kind)
    return stamp, value


def inspect_bag(folder):
    db = next(folder.glob('*.db3'))
    connection = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    topics = {row[0]: (row[1], row[2]) for row in connection.execute(
        'SELECT id, name, type FROM topics')}
    event_counts = Counter()
    values = {'front': [], 'rear': [], 'command': []}
    latest = {'front': None, 'rear': None, 'command': None}
    scored = []
    references = {'master': [], 'rover': []}
    observer = Observer()
    skipped = 0
    for recorded_ns, topic_id, blob in connection.execute(
            'SELECT timestamp, topic_id, data FROM messages ORDER BY timestamp'):
        received_at = recorded_ns * 1e-9
        name, kind = topics[topic_id]
        event_counts[name] += 1
        if name.endswith('/master/vel') or name.endswith('/rover/vel'):
            _, xyz = decode(blob, kind)
            receiver = 'master' if '/master/' in name else 'rover'
            references[receiver].append((received_at, math.sqrt(sum(x * x for x in xyz))))
            continue
        if name == '/vehicle/driver_position_cmd':
            _, value = decode(blob, kind)
            latest['command'] = (received_at, value)
            values['command'].append(value)
            continue
        if name not in ('/vehicle/front_bogie_velocity', '/vehicle/rear_bogie_velocity'):
            continue
        side = 'front' if 'front_' in name else 'rear'
        _, value = decode(blob, kind)
        t = received_at
        latest[side] = (t, value)
        values[side].append(value)
        if observer.time is not None and t <= observer.time:
            skipped += 1
            continue

        def snapshot(key, timeout):
            sample = latest[key]
            if sample is None or sample[0] > t:
                return None, timeout + 1
            return sample[1], t - sample[0]

        command, command_age = snapshot('command', observer.cfg.command_timeout)
        front, front_age = snapshot('front', observer.cfg.velocity_timeout)
        rear, rear_age = snapshot('rear', observer.cfg.velocity_timeout)
        result = observer.step(t, command, front, rear, command_age=command_age,
                               front_age=front_age, rear_age=rear_age)
        available = [value for key in ('front', 'rear')
                     if (saved := latest[key]) is not None
                     and 0 <= t - saved[0] <= observer.cfg.velocity_timeout
                     and math.isfinite(value := saved[1])]
        raw = (statistics.median(available) * observer.cfg.wheel_speed_scale
               if available else None)
        scored.append((t, result.velocity, result.distance, result.status, raw))
    connection.close()
    reference = references['master'] or references['rover']
    # Match the nearest reference after replay. This has no effect on estimates.
    errors = []
    raw_errors = []
    truth_speeds = []
    statuses = Counter()
    ref_index = 0
    for t, v, _, status, raw in scored:
        statuses[status] += 1
        while ref_index + 1 < len(reference) and abs(reference[ref_index + 1][0] - t) < abs(reference[ref_index][0] - t):
            ref_index += 1
        if reference and abs(reference[ref_index][0] - t) <= 0.05:
            truth = reference[ref_index][1]
            truth_speeds.append(truth)
            errors.append(v - truth)
            if raw is not None:
                raw_errors.append(raw - truth)
    def distribution(items):
        finite = [x for x in items if math.isfinite(x)]
        return {'count': len(items), 'min': min(finite) if finite else None,
                'median': statistics.median(finite) if finite else None,
                'max': max(finite) if finite else None}
    return {
        'bag': folder.name,
        'messages': dict(event_counts),
        'inputs': {key: distribution(value) for key, value in values.items()},
        'estimates': len(scored),
        'statuses': dict(statuses),
        'out_of_order_stamps': skipped,
        'gnss_speed_pairs': len(errors),
        'gnss_reference': 'master' if references['master'] else 'rover' if references['rover'] else None,
        'gnss_speed_mps': distribution(truth_speeds),
        'speed_rmse_mps': math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
        'raw_wheel_rmse_mps': math.sqrt(sum(e * e for e in raw_errors) / len(raw_errors)) if raw_errors else None,
        'speed_bias_mps': statistics.mean(errors) if errors else None,
        'final_relative_distance_m': scored[-1][2] if scored else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path, help='folder containing extracted bag directories')
    parser.add_argument('--bag', action='append', help='bag ID, repeatable; omit for all')
    parser.add_argument('--output', type=Path, help='write JSON report to this file')
    args = parser.parse_args()
    folders = sorted(args.data.glob('*/metadata.yaml'))
    if args.bag:
        folders = [p for p in folders if p.parent.name in set(args.bag)]
    if not folders:
        parser.error('no matching bags')
    results = [inspect_bag(p.parent) for p in folders]
    report = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    else:
        print(report)


if __name__ == '__main__':
    main()
