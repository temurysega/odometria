"""Build a static route from a training bag's GNSS fixes, offline only."""
import argparse
import json
import math
from pathlib import Path
import sqlite3

from analyze_dataset import decode


def build(bag, output, *, window=31, spacing=2.0):
    db = next(bag.glob('*.db3'))
    connection = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    row = connection.execute("SELECT id, type FROM topics WHERE name='/sensing/gnss/master/fix'").fetchone()
    if row is None:
        raise ValueError('training bag has no master GNSS fixes')
    topic_id, kind = row
    fixes = [decode(blob, kind)[1] for (blob,) in connection.execute(
        'SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (topic_id,))]
    connection.close()
    if len(fixes) < window:
        raise ValueError('training bag is too short')
    prefix = [[0.0] for _ in range(3)]
    for point in fixes:
        for axis in range(3):
            prefix[axis].append(prefix[axis][-1] + point[axis])
    smoothed = []
    half = window // 2
    for index in range(len(fixes)):
        left, right = max(0, index - half), min(len(fixes), index + half + 1)
        smoothed.append(tuple((prefix[axis][right] - prefix[axis][left]) /
                              (right - left) for axis in range(3)))
    lat0 = smoothed[0][0]
    cos_lat = math.cos(math.radians(lat0))

    def separation(a, b):
        east = 6371000 * math.radians(a[1] - b[1]) * cos_lat
        north = 6371000 * math.radians(a[0] - b[0])
        return math.hypot(east, north)

    kept = [smoothed[0]]
    for point in smoothed[1:-1]:
        if separation(point, kept[-1]) >= spacing:
            kept.append(point)
    if separation(smoothed[-1], kept[-1]) > 0.2:
        kept.append(smoothed[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'source_bag': bag.name, 'points':
                      [[round(v, 9) for v in point] for point in kept]},
                      separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'{len(fixes)} fixes -> {len(kept)} route points: {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.bag, args.output)


if __name__ == '__main__':
    main()
