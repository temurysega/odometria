"""Офлайн проигрывание bag через ядро узла и сравнение с GNSS эталоном.

Сообщения подаются в порядке времени записи, как при ros2 bag play.
GNSS фиксы передаются ядру все, но оно принимает только окно начальной
выставки; скорость GNSS в ядро не попадает вовсе.

Эталон:
  скорость: модуль горизонтальной скорости /sensing/gnss/master/vel
            (rover/vel, если master нет);
  положение: /sensing/gnss/master/fix в ENU WGS84 от первого валидного фикса.
Сопоставление по header.stamp с допуском 0,05 с в двух вариантах:
  out2ref  для каждого выхода ближайшая эпоха эталона;
  ref2out  для каждой эпохи эталона ближайший выход.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'odometria'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bagdata import load, unique_bags  # noqa: E402
from build_map import to_enu  # noqa: E402
from odometria.core import CoreConfig, OdometryCore  # noqa: E402
from odometria.observer import ObserverConfig  # noqa: E402
from odometria.track import TrackMap  # noqa: E402
from odometria.traction import TractionModel  # noqa: E402

TOL = 0.05


def stream(data):
    events = []
    for side in ('front', 'rear'):
        if side in data:
            for rec, stamp, value in data[side][:, :3]:
                events.append((rec, 0, side, stamp, value))
    if 'cmd' in data:
        for rec, stamp, value in data['cmd'][:, :3]:
            events.append((rec, 1, 'cmd', stamp, int(value)))
    for src in ('master', 'rover'):
        key = f'{src}_fix'
        if key in data:
            for row in data[key]:
                events.append((row[0], 2, src, row[1], (row[2], row[3], row[4], int(row[5]))))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def run_core(data, core):
    outputs = []
    timing = []
    for _, kind, name, stamp, value in stream(data):
        begin = time.perf_counter()
        if kind == 0:
            res = core.on_wheel(name, stamp, value)
        elif kind == 1:
            res = core.on_command(stamp, value)
        else:
            core.on_fix(name, stamp, *value)
            res = []
        timing.append(time.perf_counter() - begin)
        outputs.extend(res)
    return outputs, np.asarray(timing)


def reference(data):
    vel = data.get('master_vel')
    if vel is None:
        vel = data.get('rover_vel')
    src = 'master' if 'master_fix' in data else ('rover' if 'rover_fix' in data else None)
    ref = {'vel': None, 'pos': None, 'src': src}
    if vel is not None and len(vel):
        offset = vel[:, 1] - vel[:, 0]
        # сбои эталона: скачок метки времени GNSS на секунду
        trusted = np.abs(offset - np.median(offset)) < 0.3
        ref['vel'] = np.c_[vel[:, 1], np.hypot(vel[:, 2], vel[:, 3]), trusted]
    if src is not None:
        fix = data[f'{src}_fix']
        ok = (fix[:, 5] >= 0) & np.isfinite(fix[:, 2])
        fix = fix[ok]
        if len(fix):
            enu = to_enu(fix[:, 2], fix[:, 3], fix[:, 4], origin=tuple(fix[0, 2:5]))
            offset = fix[:, 1] - fix[:, 0]
            clean = (fix[:, 5] == 2) & (np.abs(offset - np.median(offset)) < 0.3)
            ref['pos'] = np.c_[fix[:, 1], enu, clean]
    return ref


def match(t_a, t_b):
    """Индекс ближайшего t_b для каждого t_a и признак попадания в допуск."""
    idx = np.clip(np.searchsorted(t_b, t_a), 1, len(t_b) - 1)
    left = t_b[idx - 1]
    right = t_b[idx]
    idx = np.where(np.abs(t_a - left) <= np.abs(right - t_a), idx - 1, idx)
    return idx, np.abs(t_b[idx] - t_a) <= TOL + 1e-9


def metrics(outputs, ref, scheme='out2ref'):
    if not outputs:
        return {}
    o_t = np.array([o.stamp for o in outputs])
    order = np.argsort(o_t, kind='stable')
    o_t = o_t[order]
    o_v = np.array([outputs[i].velocity for i in order])
    o_p = np.array([outputs[i].position for i in order])
    res = {}
    if ref['vel'] is not None:
        rt, rv = ref['vel'][:, 0], ref['vel'][:, 1]
        rtrust = ref['vel'][:, 2] > 0
        if scheme == 'out2ref':
            j, ok = match(o_t, rt)
            e = o_v[ok] - rv[j[ok]]
            truth = rv[j[ok]]
            trust = rtrust[j[ok]]
        else:
            j, ok = match(rt, o_t)
            e = o_v[j[ok]] - rv[ok]
            truth = rv[ok]
            trust = rtrust[ok]
        # провал GNSS: эталон ноль при движении
        trust = trust & ~((truth < 0.05) & (np.abs(e) > 1.0))
        if len(e):
            acc = np.abs(np.gradient(truth)) / 0.1 if len(truth) > 2 else np.zeros(len(truth))
            trans = acc > 0.3
            stop = truth < 0.1
            res.update({
                'v_pairs': int(len(e)), 'v_rmse': float(np.sqrt(np.mean(e ** 2))),
                'v_mae': float(np.mean(np.abs(e))), 'v_bias': float(np.mean(e)),
                'v_rmse_transient': float(np.sqrt(np.mean(e[trans] ** 2))) if trans.any() else None,
                'v_bias_transient': float(np.mean(e[trans])) if trans.any() else None,
                'v_rmse_stop': float(np.sqrt(np.mean(e[stop] ** 2))) if stop.any() else None,
                'v_sq_sum': float(np.sum(e ** 2)), 'v_abs_sum': float(np.sum(np.abs(e))),
                'v_p99_abs': float(np.percentile(np.abs(e), 99)),
                'v_rmse_clean': float(np.sqrt(np.mean(e[trust] ** 2))) if trust.any() else None,
                'v_sq_sum_clean': float(np.sum(e[trust] ** 2)), 'v_pairs_clean': int(trust.sum()),
            })
    if ref['pos'] is not None:
        rt = ref['pos'][:, 0]
        rp = ref['pos'][:, 1:4]
        clean = ref['pos'][:, 4] > 0
        if scheme == 'out2ref':
            j, ok = match(o_t, rt)
            est = o_p[ok]
            truth = rp[j[ok]]
            cl = clean[j[ok]]
        else:
            j, ok = match(rt, o_t)
            est = o_p[j[ok]]
            truth = rp[ok]
            cl = clean[ok]
        if len(est) > 10:
            err = est - truth
            d3 = np.linalg.norm(err, axis=1)
            d2 = np.linalg.norm(err[:, :2], axis=1)
            # направление движения по эталону для разложения ошибки
            tang = np.gradient(truth[:, :2], axis=0)
            for _ in range(3):
                tang = 0.25 * np.roll(tang, 1, 0) + 0.5 * tang + 0.25 * np.roll(tang, -1, 0)
            norm = np.linalg.norm(tang, axis=1)
            moving = norm > 1e-3
            tang[moving] /= norm[moving, None]
            along = np.sum(err[:, :2] * tang, axis=1)
            cross = err[:, 0] * tang[:, 1] - err[:, 1] * tang[:, 0]
            path = float(np.sum(np.linalg.norm(np.diff(truth[cl][::10, :2], axis=0), axis=1))) if cl.sum() > 20 else float('nan')
            res.update({
                'p_pairs': int(len(err)), 'p_rmse_3d': float(np.sqrt(np.mean(d3 ** 2))),
                'p_rmse_2d': float(np.sqrt(np.mean(d2 ** 2))), 'p_mean_3d': float(np.mean(d3)),
                'p_max_3d': float(np.max(d3)),
                'p_rmse_3d_clean': float(np.sqrt(np.mean(d3[cl] ** 2))) if cl.any() else None,
                'along_rmse': float(np.sqrt(np.mean(along[moving] ** 2))) if moving.any() else None,
                'along_mean': float(np.mean(np.abs(along[moving]))) if moving.any() else None,
                'along_max': float(np.max(np.abs(along[moving]))) if moving.any() else None,
                'cross_rmse': float(np.sqrt(np.mean(cross[moving] ** 2))) if moving.any() else None,
                'z_rmse': float(np.sqrt(np.mean(err[:, 2] ** 2))),
                'final_err_3d': float(d3[-1]), 'path_m': path,
                'drift_pct': float(100 * d3[-1] / path) if path and path > 100 else None,
                'p_sq_sum': float(np.sum(d3 ** 2)),
            })
    return res


def evaluate(args):
    folder, cache, map_path, overrides = args
    data = load(folder, cache)
    track = TrackMap.load(map_path) if map_path else None
    cfg = CoreConfig(**overrides.get('core', {}))
    ocfg = ObserverConfig(**overrides.get('observer', {}))
    core = OdometryCore(cfg, ocfg, track_map=track, model=TractionModel.load())
    outputs, timing = run_core(data, core)
    ref = reference(data)
    duration = data['cmd'][-1, 0] - data['cmd'][0, 0] if 'cmd' in data else 0.0
    res = {'bag': folder.name, 'duration_s': float(duration), 'outputs': len(outputs),
           'rate_hz': len(outputs) / duration if duration > 0 else None,
           'step_ms_p50': float(np.percentile(timing, 50) * 1e3),
           'step_ms_p99': float(np.percentile(timing, 99) * 1e3),
           'step_ms_max': float(timing.max() * 1e3),
           'landmarks': core.counters.landmarks, 'landmark_skipped': core.counters.landmark_skipped,
           'scale_k': core.localizer.scale, 'map_matched': core.localizer.start_s is not None,
           'reference': ref['src']}
    for scheme in ('out2ref', 'ref2out'):
        for k, v in metrics(outputs, ref, scheme).items():
            res[f'{scheme}.{k}'] = v
    return res


def summarize(rows, scheme='out2ref'):
    def col(name):
        return [r[f'{scheme}.{name}'] for r in rows if r.get(f'{scheme}.{name}') is not None]
    out = {}
    vp = sum(col('v_pairs'))
    if vp:
        out['v_rmse_pooled'] = math.sqrt(sum(col('v_sq_sum')) / vp)
        out['v_mae_pooled'] = sum(col('v_abs_sum')) / vp
        out['v_rmse_median_bag'] = float(np.median(col('v_rmse')))
        out['v_rmse_worst_bag'] = float(np.max(col('v_rmse')))
        out['v_bias_mean'] = float(np.mean(col('v_bias')))
        out['v_rmse_pooled_clean'] = math.sqrt(sum(col('v_sq_sum_clean')) / max(1, sum(col('v_pairs_clean'))))
        out['v_rmse_worst_bag_clean'] = float(np.max(col('v_rmse_clean')))
    pp = sum(col('p_pairs'))
    if pp:
        out['p_rmse3d_pooled'] = math.sqrt(sum(col('p_sq_sum')) / pp)
        out['p_rmse3d_clean_median'] = float(np.median(col('p_rmse_3d_clean')))
        out['p_rmse3d_median_bag'] = float(np.median(col('p_rmse_3d')))
        out['p_rmse3d_worst_bag'] = float(np.max(col('p_rmse_3d')))
        out['along_rmse_median'] = float(np.median(col('along_rmse')))
        out['cross_rmse_median'] = float(np.median(col('cross_rmse')))
        out['drift_pct_median'] = float(np.median(col('drift_pct'))) if col('drift_pct') else None
        out['final_err_median'] = float(np.median(col('final_err_3d')))
    out['bags'] = len(rows)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--bag', action='append')
    parser.add_argument('--map', type=Path, default=ROOT / 'src/odometria/odometria/data/track_map.json')
    parser.add_argument('--fold-maps', nargs=2, type=Path, metavar=('MAP_A', 'MAP_B'),
                        help='карты без дней A и без дней B для честной отложенной проверки')
    parser.add_argument('--day-maps', type=Path,
                        help='каталог с картами map_<день>.json, построенными без этого дня')
    parser.add_argument('--no-map', action='store_true')
    parser.add_argument('--set', action='append', default=[],
                        help='переопределение параметра, например core.use_landmarks=False')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    overrides = {'core': {}, 'observer': {}}
    for item in args.set:
        key, value = item.split('=', 1)
        group, name = key.split('.', 1)
        overrides[group][name] = json.loads(value.lower() if value in ('True', 'False') else value)
    folders = unique_bags(args.data, args.cache)
    if args.bag:
        folders = [f for f in folders if f.name in set(args.bag)]
    jobs = []
    for folder in folders:
        if args.no_map:
            map_path = None
        elif args.day_maps:
            map_path = str(args.day_maps / f'map_{day_of(folder.name, args.cache, args.data)}.json')
        elif args.fold_maps:
            map_path = str(args.fold_maps[0] if fold_of(folder.name, args.cache, args.data) == 'A' else args.fold_maps[1])
        else:
            map_path = str(args.map)
        jobs.append((folder, args.cache, map_path, overrides))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(evaluate, jobs))
    report = {'summary_out2ref': summarize(rows, 'out2ref'),
              'summary_ref2out': summarize(rows, 'ref2out'), 'bags': rows,
              'overrides': overrides,
              'map': 'day' if args.day_maps else 'fold' if args.fold_maps else ('none' if args.no_map else 'full')}
    print(json.dumps({k: report[k] for k in ('summary_out2ref', 'summary_ref2out')}, ensure_ascii=False, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


FOLD_A_DAYS = ('2026-05-05', '2026-08-10', '2026-09-03')


def day_of(name, cache, root):
    import datetime
    data = load(Path(root) / name, cache)
    return datetime.datetime.fromtimestamp(float(data['cmd'][0, 0]), datetime.timezone.utc).date().isoformat()


def fold_of(name, cache, root):
    """Разбиение по дням записи: A = 5 мая, 10 августа, 3 сентября; B = остальные."""
    import datetime
    data = load(Path(root) / name, cache)
    day = datetime.datetime.fromtimestamp(float(data['cmd'][0, 0]), datetime.timezone.utc).date().isoformat()
    return 'A' if day in FOLD_A_DAYS else 'B'


if __name__ == '__main__':
    main()
