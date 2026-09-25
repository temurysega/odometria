"""Inspect GNSS trajectory geometry in one supplied bag (offline only)."""
import argparse
import math
from pathlib import Path
import sqlite3

from analyze_dataset import decode


def inspect(folder):
    db = next(folder.glob('*.db3'))
    connection = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    topics = {i: (name, kind) for i, name, kind in connection.execute(
        'SELECT id, name, type FROM topics')}
    fixes = {'master': [], 'rover': []}
    velocities = {'master': [], 'rover': []}
    for received_ns, topic_id, blob in connection.execute(
            'SELECT timestamp, topic_id, data FROM messages ORDER BY timestamp'):
        name, kind = topics[topic_id]
        if '/sensing/gnss/' not in name:
            continue
        source = 'master' if '/master/' in name else 'rover'
        _, value = decode(blob, kind)
        (fixes if name.endswith('/fix') else velocities)[source].append(
            (received_ns * 1e-9, value))
    connection.close()
    source = 'master' if fixes['master'] else 'rover'
    points = fixes[source]
    speed = velocities[source]
    if not points:
        return {'bag': folder.name, 'fixes': 0}
    lat0, lon0, alt0 = points[0][1]
    radius = 6371000.0
    antenna = None
    if fixes['master'] and fixes['rover']:
        mlat, mlon, malt = fixes['master'][0][1]
        rlat, rlon, ralt = fixes['rover'][0][1]
        antenna = (radius * math.radians(rlon - mlon) * math.cos(math.radians(mlat)),
                   radius * math.radians(rlat - mlat), ralt - malt)
    xy = [(radius * math.radians(lon - lon0) * math.cos(math.radians(lat0)),
           radius * math.radians(lat - lat0), alt - alt0)
          for _, (lat, lon, alt) in points]
    path = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    first_five = next((i for i, (t, _) in enumerate(points) if t - points[0][0] >= 5),
                      len(points) - 1)
    v_integral = [0.0, 0.0, 0.0]
    for (ta, va), (tb, vb) in zip(speed, speed[1:]):
        dt = tb - ta
        if 0 < dt <= 1:
            for axis in range(3):
                v_integral[axis] += (va[axis] + vb[axis]) * dt / 2
    return {'bag': folder.name, 'source': source, 'fixes': len(points),
            'duration_s': points[-1][0] - points[0][0],
            'net_enu_m': xy[-1], 'path_length_m': path,
            'range_e_m': (min(p[0] for p in xy), max(p[0] for p in xy)),
            'range_n_m': (min(p[1] for p in xy), max(p[1] for p in xy)),
            'first_5s_displacement_m': xy[first_five],
            'integrated_velocity_xyz_m': v_integral,
            'rover_minus_master_enu_m': antenna,
            'first_fix': (lat0, lon0, alt0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path, nargs='+')
    args = parser.parse_args()
    for bag in args.bag:
        print(inspect(bag))


if __name__ == '__main__':
    main()
