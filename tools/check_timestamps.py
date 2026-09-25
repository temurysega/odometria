"""Check output timestamp alignment against GNSS headers on one recorded bag."""
import argparse
from bisect import bisect_left
from pathlib import Path
import sqlite3

from analyze_dataset import decode


def nearest_error(stamps, value):
    index = bisect_left(stamps, value)
    return min((abs(stamps[i] - value) for i in (index - 1, index)
                if 0 <= i < len(stamps)), default=float('inf'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path)
    args = parser.parse_args()
    db = next(args.bag.glob('*.db3'))
    connection = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    topics = {i: (n, kind) for i, n, kind in connection.execute(
        'SELECT id, name, type FROM topics')}
    wheels = []
    master = []
    rover = []
    for recorded, topic_id, blob in connection.execute(
            'SELECT timestamp, topic_id, data FROM messages ORDER BY timestamp'):
        name, kind = topics[topic_id]
        if name in ('/vehicle/front_bogie_velocity', '/vehicle/rear_bogie_velocity',
                    '/sensing/gnss/master/vel', '/sensing/gnss/rover/vel'):
            stamp, _ = decode(blob, kind)
            if '/vehicle/' in name:
                wheels.append((recorded * 1e-9, stamp, 'front' if '/front_' in name else 'rear'))
            elif '/master/' in name:
                master.append(stamp)
            else:
                rover.append(stamp)
    connection.close()
    reference = sorted(master or rover)
    if not reference:
        parser.error('bag has no GNSS velocity headers')
    anchor = None
    matches = {'front': [0, 0], 'rear': [0, 0]}
    backwards = 0
    last_output = None
    outputs = []
    for received, stamp, side in wheels:
        if side == 'front':
            anchor = (received, stamp)
            output = stamp
        elif anchor:
            output = anchor[1] + received - anchor[0]
        else:
            output = stamp
        matches[side][1] += 1
        matches[side][0] += nearest_error(reference, output) <= 0.05
        outputs.append(output)
        if last_output is not None and output < last_output:
            backwards += 1
        last_output = output
    outputs.sort()
    reference_matched = sum(nearest_error(outputs, value) <= 0.05 for value in reference)
    print(args.bag.name, 'reference', 'master' if master else 'rover',
          'matched_outputs', matches, 'matched_reference',
          (reference_matched, len(reference)), 'backwards', backwards)


if __name__ == '__main__':
    main()
