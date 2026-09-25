"""Офлайн построение карты пути и ориентиров остановок по обучающим записям.

GNSS читается только здесь, при подготовке карты. Узел получает готовый
файл и в рабочем контуре GNSS не использует.

Координаты карты: плоские MGRS (UTM зона 37 минус угол квадрата сетки
300 000 / 6 100 000 м), высота антенны. В этой же системе заданы карты
организаторов и, по ответу экспертов, эталон.

Шаги:
1. RTK фиксы master антенны переводятся в MGRS по эллипсоиду WGS84.
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
MGRS_ZONE = 37
MGRS_ORIGIN = (300000.0, 6100000.0)
# платформа восточного кольца в MGRS
EAST_PLATFORM = np.array([103632.1, 86047.6])
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


def to_mgrs(lat, lon, alt, zone=MGRS_ZONE, origin=MGRS_ORIGIN):
    """Векторный UTM (ряды Крюгера) минус угол квадрата сетки MGRS."""
    n = F / (2.0 - F)
    big_a = A / (1.0 + n) * (1.0 + n ** 2 / 4.0 + n ** 4 / 64.0)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180,
             13 * n ** 2 / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440,
             61 * n ** 3 / 240 - 103 * n ** 4 / 140,
             49561 * n ** 4 / 161280)
    phi = np.radians(lat)
    dlon = np.radians(lon) - np.radians(6.0 * zone - 183.0)
    c = 2.0 * np.sqrt(n) / (1.0 + n)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi = np.arctan2(t, np.cos(dlon))
    eta = np.arctanh(np.sin(dlon) / np.sqrt(1.0 + t * t))
    east, north = eta.copy(), xi.copy()
    for j, a in enumerate(alpha, start=1):
        east = east + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        north = north + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return np.stack([500000.0 + 0.9996 * big_a * east - origin[0],
                     0.9996 * big_a * north - origin[1], np.asarray(alt, dtype=float)], -1)


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
    return {'t': fix[:, 1], 'E': to_mgrs(fix[:, 2], fix[:, 3], fix[:, 4]),
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


def load_official(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    return np.array([[p['x'], p['y'], p['z']] for p in doc['points']])[doc['paths'][0]['point_indices']]


def splice_official(line, official):
    """Основной ход берётся из карты организаторов, концы у колец из своих данных.

    Карта организаторов описывает base_link (на 3,1 м ниже антенн), поэтому
    её высота поднимается до уровня антенны по медиане на общем участке.
    """
    s, lat, dist = project(line, official)
    near = dist < 1.0
    lift = float(np.median(line[np.clip(np.round(s[near]).astype(int), 0, len(line) - 1), 2] - official[near, 2]))
    s0, s1 = s[0], s[-1]
    if dist[0] > 1.5 or dist[-1] > 1.5 or s1 <= s0:
        raise ValueError('карта организаторов не стыкуется с линией')
    head = line[:int(np.floor(s0))]
    tail = line[int(np.ceil(s1)) + 1:]
    body = official + np.array([0.0, 0.0, lift])
    return resample(np.vstack([head, body, tail])), lift, float(s0), float(s1)


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


def official_check(circuit, paths):
    """Сверка с картами организаторов: поперечное отклонение и разница высот."""
    out = []
    for path in paths:
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
        pts = np.array([[p['x'], p['y'], p['z']] for p in doc['points']])[doc['paths'][0]['point_indices']]
        s, lat, dist = project(circuit, pts)
        ok = dist < 5.0
        idx = np.clip(np.round(s[ok]).astype(int), 0, len(circuit) - 1)
        dz = circuit[idx, 2] - pts[ok, 2]
        out.append({'map': Path(path).stem, 'points': int(len(pts)), 'covered_share': round(float(ok.mean()), 4),
                    'lateral_abs_median_m': round(float(np.median(np.abs(lat[ok]))), 3),
                    'lateral_abs_p95_m': round(float(np.percentile(np.abs(lat[ok]), 95)), 3),
                    'antenna_minus_map_z_median_m': round(float(np.median(dz)), 3)})
        print('сверка', out[-1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path, help='каталог с распакованными bag')
    parser.add_argument('--cache', type=Path, default=None)
    parser.add_argument('--exclude', default='', help='список bag через запятую')
    parser.add_argument('--exclude-days', default='', help='даты записи через запятую, например 2026-05-05')
    parser.add_argument('--official', type=Path, nargs='*', default=[],
                        help='карты организаторов (json с points и paths) для сверки')
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
    spliced = []
    for path in args.official:
        official = load_official(path)
        # направление карты по её началу: западный конец значит рейс на восток
        side = 'E' if official[0, 0] < official[-1, 0] else 'W'
        lines[side], lift, s0, s1 = splice_official(lines[side], official)
        spliced.append({'map': Path(path).stem, 'direction': side, 'lift_m': round(lift, 3),
                        'replaced_s': [round(s0, 1), round(s1, 1)]})
        print(f'{side}: основной ход {s1 - s0:.0f} м из карты организаторов, подъём высоты {lift:.3f} м')
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
    check = {'spliced': spliced, 'agreement': official_check(circuit, args.official)} if args.official else None
    doc = {
        'description': 'Траектория master антенны по RTK фиксам (MGRS, высота антенны) и ориентиры остановок',
        'built': datetime.date.today().isoformat(),
        'frame': {'type': 'MGRS', 'utm_zone': MGRS_ZONE, 'grid_origin': list(MGRS_ORIGIN)},
        'official_map_check': check,
        'step_m': 1.0,
        'east_junction_s': float(len(lines['E'])),
        'closed': closed,
        'points': [[round(float(x), 3) for x in p] for p in circuit],
        'stops': marks,
        'excluded_bags': sorted(exclude),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('записано', args.output)


if __name__ == '__main__':
    main()
