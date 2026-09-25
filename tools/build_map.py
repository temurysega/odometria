"""Офлайн построение карты пути и ориентиров остановок по обучающим записям.

GNSS читается только здесь, при подготовке карты. Узел получает готовый
файл и в рабочем контуре GNSS не использует.

Шаги:
1. RTK фиксы master антенны переводятся в ENU карты по эллипсоиду WGS84.
2. Для каждого направления берётся опорная поездка, фиксы усредняются по
   пройденному колёсами пути, выбросы режутся скользящей медианой.
3. Осевая уточняется медианой поперечных отклонений и высот всех поездок.
4. Два направления сшиваются у восточного кольца в один контур.
5. Места остановок кластеризуются вдоль контура и становятся ориентирами.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial import cKDTree

from bagdata import load, unique_bags

MAP_ORIGIN = (55.810367065, 37.462266845, 150.0)
WEST_TERMINAL = np.array([-4565.0, -1195.0])
EAST_PLATFORM = np.array([2.0, 3.5])
A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)


def ecef(lat, lon, alt):
    la = np.radians(lat)
    lo = np.radians(lon)
    n = A / np.sqrt(1.0 - E2 * np.sin(la) ** 2)
    return np.stack([(n + alt) * np.cos(la) * np.cos(lo),
                     (n + alt) * np.cos(la) * np.sin(lo),
                     (n * (1.0 - E2) + alt) * np.sin(la)], -1)


def rotation(lat0, lon0):
    la = np.radians(lat0)
    lo = np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0.0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def to_enu(lat, lon, alt, origin=MAP_ORIGIN):
    return (ecef(lat, lon, alt) - ecef(*origin)) @ rotation(*origin[:2]).T


def wheel_distance(data):
    f, r = data['front'], data['rear']
    t = np.unique(np.r_[f[:, 1], r[:, 1]])
    v = 0.5 * (np.interp(t, f[:, 1], f[:, 2]) + np.interp(t, r[:, 1], r[:, 2])) / 3.6
    return t, v, np.r_[0.0, np.cumsum(np.diff(t) * 0.5 * (v[1:] + v[:-1]))]


def rtk_trip(data):
    fix = data.get('master_fix')
    if fix is None or len(fix) < 100:
        return None
    offset = fix[:, 1] - fix[:, 0]
    ok = (fix[:, 5] == 2) & np.isfinite(fix[:, 2]) & (np.abs(offset - np.median(offset)) < 0.3)
    fix = fix[ok]
    if len(fix) < 100:
        return None
    t, _, dist = wheel_distance(data)
    return {'t': fix[:, 1], 'E': to_enu(fix[:, 2], fix[:, 3], fix[:, 4]),
            'D': np.interp(fix[:, 1], t, dist), 'rtk_share': ok.mean()}


def direction(data):
    fix = data.get('master_fix')
    if fix is None:
        fix = data.get('rover_fix')
    if fix is None or len(fix) < 100:
        return None
    e = to_enu(fix[[0, -1], 2], fix[[0, -1], 3], fix[[0, -1], 4])
    east_start = np.hypot(*e[0, :2]) < 200
    east_end = np.hypot(*e[1, :2]) < 200
    west_start = np.hypot(*(e[0, :2] - WEST_TERMINAL)) < 400
    west_end = np.hypot(*(e[1, :2] - WEST_TERMINAL)) < 400
    if east_start and west_end:
        return 'W'
    if west_start and east_end:
        return 'E'
    return None


def resample(points, step=1.0):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, seg > 1e-3]]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    grid = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(grid, s, points[:, i]) for i in range(3)], 1)


def seed_line(trip):
    """Опорная линия: медиана фиксов по метрам колёсного пути и срез выбросов."""
    bins = np.round(trip['D']).astype(int)
    count = bins.max() + 1
    line = np.full((count, 3), np.nan)
    order = np.argsort(bins)
    sorted_bins, points = bins[order], trip['E'][order]
    starts = np.searchsorted(sorted_bins, np.arange(count))
    ends = np.searchsorted(sorted_bins, np.arange(count), side='right')
    for j in range(count):
        if ends[j] > starts[j]:
            line[j] = np.median(points[starts[j]:ends[j]], axis=0)
    idx = np.arange(count)
    good = ~np.isnan(line[:, 0])
    for i in range(3):
        line[:, i] = np.interp(idx, idx[good], line[good, i])
    for _ in range(2):
        med = np.stack([median_filter(line[:, i], size=31, mode='nearest') for i in range(3)], 1)
        bad = np.linalg.norm(line[:, :2] - med[:, :2], axis=1) > 1.0
        line[bad] = med[bad]
    line = np.stack([gaussian_filter1d(line[:, 0], 2, mode='nearest'),
                     gaussian_filter1d(line[:, 1], 2, mode='nearest'),
                     gaussian_filter1d(line[:, 2], 8, mode='nearest')], 1)
    return resample(line)


def tangents(line):
    t = np.gradient(line[:, :2], axis=0)
    return t / np.linalg.norm(t, axis=1)[:, None]


def project(line, points):
    """Проекция точек на ломаную: дуговая координата, поперечное смещение, расстояние."""
    tree = cKDTree(line[:, :2])
    _, idx = tree.query(points[:, :2])
    idx = np.clip(idx, 0, len(line) - 2)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1))]
    best_s = np.zeros(len(points))
    best_lat = np.zeros(len(points))
    best_d = np.full(len(points), np.inf)
    for shift in (-1, 0):
        i = np.clip(idx + shift, 0, len(line) - 2)
        a, b = line[i, :2], line[i + 1, :2]
        ab = b - a
        l2 = np.maximum((ab ** 2).sum(1), 1e-9)
        u = np.clip(((points[:, :2] - a) * ab).sum(1) / l2, 0, 1)
        q = a + u[:, None] * ab
        dist = np.linalg.norm(points[:, :2] - q, axis=1)
        cross = (ab[:, 0] * (points[:, 1] - a[:, 1]) - ab[:, 1] * (points[:, 0] - a[:, 0])) / np.sqrt(l2)
        better = dist < best_d
        best_d[better] = dist[better]
        best_s[better] = (arc[i] + u * np.sqrt(l2))[better]
        best_lat[better] = cross[better]
    return best_s, best_lat, best_d


def refine(line, trips, iterations=3, max_lateral=2.0):
    for _ in range(iterations):
        n = len(line)
        s_all, lat_all, alt_all = [], [], []
        for trip in trips:
            s, lat, _ = project(line, trip['E'])
            ok = np.abs(lat) < max_lateral
            s_all.append(s[ok])
            lat_all.append(lat[ok])
            alt_all.append(trip['E'][ok, 2])
        s_all = np.concatenate(s_all)
        lat_all = np.concatenate(lat_all)
        alt_all = np.concatenate(alt_all)
        bins = np.clip(np.round(s_all).astype(int), 0, n - 1)
        order = np.argsort(bins)
        bins, lat_all, alt_all = bins[order], lat_all[order], alt_all[order]
        starts = np.searchsorted(bins, np.arange(n))
        ends = np.searchsorted(bins, np.arange(n), side='right')
        lat = np.full(n, np.nan)
        alt = np.full(n, np.nan)
        for j in range(n):
            if ends[j] - starts[j] >= 3:
                lat[j] = np.median(lat_all[starts[j]:ends[j]])
                alt[j] = np.median(alt_all[starts[j]:ends[j]])
        idx = np.arange(n)
        good = ~np.isnan(lat)
        lat = gaussian_filter1d(np.interp(idx, idx[good], lat[good]), 1.5, mode='nearest')
        alt = gaussian_filter1d(np.interp(idx, idx[good], alt[good]), 4, mode='nearest')
        t = tangents(line)
        normal = np.c_[-t[:, 1], t[:, 0]]
        line = resample(np.c_[line[:, :2] + lat[:, None] * normal, alt])
    return line


def join_at_east(east, west):
    """Контур: восточное направление, затем западное с точки стыковки."""
    tree = cKDTree(west[:, :2])
    dist, j = tree.query(east[-1, :2])
    if dist > 3.0:
        raise ValueError(f'направления не стыкуются у кольца: {dist:.1f} м')
    return resample(np.vstack([east, west[j + 1:]]))


def west_loop(east, west, raw):
    """Связка через западное разворотное кольцо: от конца W до начала E."""
    tree_w = cKDTree(west[-150:, :2])
    tree_e = cKDTree(east[:150, :2])
    best = None
    for name, _, trip in raw:
        pts = trip['E']
        near_w = tree_w.query(pts[:, :2])[0] < 2.0
        near_e = tree_e.query(pts[:, :2])[0] < 2.0
        for i in np.where(near_w)[0][::-1]:
            later = np.where(near_e[i:])[0]
            if len(later) and 20 < later[0] < 3000:
                seg = pts[i:i + later[0] + 1]
                if best is None or len(seg) > len(best[1]):
                    best = (name, seg)
                break
    if best is None:
        return None, None
    seg = best[1]
    keep = np.r_[True, np.linalg.norm(np.diff(seg[:, :2], axis=0), axis=1) > 0.05]
    seg = seg[keep]
    seg = np.stack([median_filter(seg[:, i], size=9, mode='nearest') for i in range(3)], 1)
    seg = np.stack([gaussian_filter1d(seg[:, 0], 3, mode='nearest'), gaussian_filter1d(seg[:, 1], 3, mode='nearest'),
                    gaussian_filter1d(seg[:, 2], 10, mode='nearest')], 1)
    return best[0], resample(seg)


def stop_events(data, trip, circuit):
    t, v, _ = wheel_distance(data)
    s, _, dist = project(circuit, trip['E'])
    ok = dist < 2.0
    ts, ss = trip['t'][ok], s[ok]
    still = v < 0.02
    events = []
    i = 1
    while i < len(t):
        if still[i] and not still[i - 1]:
            j = i
            while j < len(t) and still[j]:
                j += 1
            if t[j - 1] - t[i] >= 2.0:
                sel = (ts > t[i] + 0.3) & (ts < t[j - 1] - 0.3)
                if sel.sum() >= 3:
                    events.append(float(np.median(ss[sel])))
            i = j
        i += 1
    return events


def cluster_stops(positions, passes, gap=6.0, min_count=4):
    positions = np.sort(np.asarray(positions))
    groups = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] < gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    marks = []
    for g in groups:
        g = np.asarray(g)
        if len(g) < min_count:
            continue
        center = float(np.median(g))
        mad = float(np.median(np.abs(g - center)))
        share = len(g) / max(1, passes(center))
        marks.append({'s': round(center, 2), 'sigma': round(max(0.35, 1.4826 * mad), 2),
                      'count': int(len(g)), 'share': round(min(1.0, share), 3)})
    return marks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path, help='каталог с распакованными bag')
    parser.add_argument('--cache', type=Path, default=None)
    parser.add_argument('--exclude', default='', help='список bag через запятую')
    parser.add_argument('--exclude-days', default='', help='даты записи через запятую, например 2026-05-05')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'src/odometria/odometria/data/track_map.json')
    args = parser.parse_args()
    exclude = {x for x in args.exclude.split(',') if x}
    exclude_days = {x for x in args.exclude_days.split(',') if x}
    trips = {'W': [], 'E': []}
    raw = []
    for folder in unique_bags(args.data, args.cache):
        if folder.name in exclude:
            continue
        data = load(folder, args.cache)
        day = datetime.datetime.fromtimestamp(float(data['cmd'][0, 0]), datetime.timezone.utc).date().isoformat()
        if day in exclude_days:
            exclude.add(folder.name)
            continue
        side = direction(data)
        trip = rtk_trip(data) if side else None
        if trip is None:
            continue
        trips[side].append((folder.name, trip))
        raw.append((folder.name, data, trip))
    lines = {}
    for side in ('W', 'E'):
        # опорная поездка должна доходить до платформы восточного кольца,
        # а на западной конечной идти по самому частому пути
        west_ends = np.array([t['E'][-1 if side == 'W' else 0, :2] for _, t in trips[side]])

        def score(item):
            trip = item[1]
            end = trip['E'][0 if side == 'W' else -1, :2]
            at_terminal = np.hypot(*(end - EAST_PLATFORM)) < 6.0
            tail = trip['E'][-3000:, :2] if side == 'W' else trip['E'][:3000, :2]
            near = cKDTree(tail).query(west_ends)[0]
            popular = int(np.sum(near < 2.5))
            return (at_terminal, trip['rtk_share'] > 0.95, popular, trip['D'][-1] - trip['D'][0])
        seed_name, seed = max(trips[side], key=score)
        line = refine(seed_line(seed), [t for _, t in trips[side]])
        lines[side] = line
        print(f'{side}: опорная {seed_name}, поездок {len(trips[side])}, длина {len(line)} м')
    circuit = join_at_east(lines['E'], lines['W'])
    loop_source, loop = west_loop(lines['E'], lines['W'], raw)
    closed = False
    if loop is not None:
        # конец W и начало связки, конец связки и начало E стыкуются по ближайшим точкам
        j = cKDTree(loop[:, :2]).query(circuit[-1, :2])[1]
        k = cKDTree(loop[:, :2]).query(circuit[0, :2])[1]
        if k > j + 10:
            circuit = resample(np.vstack([circuit, loop[j + 1:k]]))
            closed = True
            print(f'западное кольцо по {loop_source}: {k - j} м, контур замкнут')
    stops = []
    covered = []
    for _, data, trip in raw:
        stops += stop_events(data, trip, circuit)
        s, _, dist = project(circuit, trip['E'])
        s = s[dist < 2.0]
        covered.append((s.min(), s.max()))
    covered = np.asarray(covered)

    def passes(s):
        return int(((covered[:, 0] <= s) & (covered[:, 1] >= s)).sum())

    marks = cluster_stops(stops, passes)
    print(f'контур {len(circuit)} м, остановок {len(stops)}, ориентиров {len(marks)}')
    doc = {
        'description': 'Осевая master антенны по RTK фиксам и ориентиры остановок',
        'built': datetime.date.today().isoformat(),
        'origin_wgs84': list(MAP_ORIGIN),
        'step_m': 1.0,
        'east_junction_s': float(len(lines['E'])),
        'closed': closed,
        'points_enu': [[round(float(x), 3) for x in p] for p in circuit],
        'stops': marks,
        'excluded_bags': sorted(exclude),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('записано', args.output)


if __name__ == '__main__':
    main()
