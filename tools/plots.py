"""Графики для README: python tools/plots.py DATA --cache CACHE"""
import argparse
import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'odometria'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bagdata import load, unique_bags  # noqa: E402
from build_map import to_enu  # noqa: E402
from odometria.core import CoreConfig, OdometryCore  # noqa: E402
from odometria.track import TrackMap  # noqa: E402
from odometria.traction import TractionModel  # noqa: E402
from replay import reference, run_core, match  # noqa: E402

OUT = ROOT / 'docs' / 'img'
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'font.size': 10.5, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.titlecolor': INK, 'legend.frameon': False, 'lines.linewidth': 2.0,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('график', OUT / name)


def plot_map():
    doc = json.loads((ROOT / 'src/odometria/odometria/data/track_map.json').read_text(encoding='utf-8'))
    p = np.asarray(doc['points_enu'])
    j = int(doc['east_junction_s'])
    tm = TrackMap.load()
    fig = plt.figure(figsize=(12, 6.2))
    ax = fig.add_axes([0.05, 0.08, 0.62, 0.84])
    ax.plot(p[:j, 0], p[:j, 1], color=BLUE, lw=1.6, label='на восток')
    ax.plot(p[j:, 0], p[j:, 1], color=ORANGE, lw=1.6, label='на запад и кольца')
    for m in tm.stops:
        x, y, _ = tm.point(m['s'])
        ax.plot(x, y, 'o', ms=4 + 8 * m['share'], mfc=SURFACE, mec=INK, mew=1.2, zorder=3)
    ax.plot([], [], 'o', ms=8, mfc=SURFACE, mec=INK, mew=1.2, label='ориентир остановки (размер: частота)')
    ax.set_aspect('equal')
    ax.set_xlabel('восток, м')
    ax.set_ylabel('север, м')
    ax.set_title(f'Карта пути: замкнутый контур {tm.length / 1000:.2f} км, {len(tm.stops)} ориентиров')
    ax.legend(loc='upper left')
    for k, (box, title) in enumerate((((-120, 30, -160, 60), 'Восточное кольцо'),
                                       ((-4680, -4440, -1260, -1090), 'Западное кольцо'))):
        sub = fig.add_axes([0.71, 0.55 - 0.47 * k, 0.27, 0.38])
        sub.plot(p[:j, 0], p[:j, 1], color=BLUE, lw=1.6)
        sub.plot(p[j:, 0], p[j:, 1], color=ORANGE, lw=1.6)
        sub.set_xlim(box[0], box[1])
        sub.set_ylim(box[2], box[3])
        sub.set_aspect('equal')
        sub.set_title(title, fontsize=10.5)
        sub.tick_params(labelsize=8)
    save(fig, 'map.png')


def plot_scale(root, cache):
    rows = []
    for folder in unique_bags(root, cache):
        d = load(folder, cache)
        g = d.get('master_vel', d.get('rover_vel'))
        if g is None or d['cmd'][-1, 0] - d['cmd'][0, 0] < 300:
            continue
        f, r = d['front'], d['rear']
        gv = np.hypot(g[:, 2], g[:, 3])
        w = 0.5 * (np.interp(g[:, 1], f[:, 1], f[:, 2]) + np.interp(g[:, 1], r[:, 1], r[:, 2])) / 3.6
        m = (gv > 1.0) & (np.abs(w - gv) < 0.3)
        day = datetime.datetime.fromtimestamp(float(d['cmd'][0, 0]), datetime.timezone.utc).date()
        rows.append((day, folder.name[:5], gv[m].sum() / w[m].sum()))
    days = sorted({r[0] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(1)
    for vehicle, color in (('30618', BLUE), ('30639', ORANGE)):
        pts = [(days.index(r[0]) + rng.uniform(-0.12, 0.12) + (0.18 if vehicle == '30639' else -0.18), (r[2] - 1) * 100)
               for r in rows if r[1] == vehicle]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o', ms=8, mfc=color, mec=SURFACE, mew=1.5,
                label=f'вагон {vehicle}')
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([d.strftime('%d.%m') for d in days])
    ax.set_ylabel('путь GNSS / путь колёс − 1, %')
    ax.set_title('Масштаб колёсных датчиков гуляет от −1,5 % до +1,6 %: нужна онлайн оценка')
    ax.legend(loc='upper left')
    save(fig, 'wheel_scale.png')


def core_on(data):
    core = OdometryCore(CoreConfig(), track_map=TrackMap.load(), model=TractionModel.load())
    outs, _ = run_core(data, core)
    return outs


def plot_slip(root, cache):
    cases = [('30618_2050d396', 396, 408, 'Юз задней тележки при торможении'),
             ('30618_33bec73f', 100, 114, 'Боксование обеих тележек при разгоне'),
             ('30618_616ec56b', 1018, 1025, 'Экстренное торможение: контроллер 0, тележки согласны'),
             ('30639_3b3d9eb8', 370, 430, 'Отказ задней тележки: ноль, затем пропуск')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for ax, (name, a, b, title) in zip(axes.ravel(), cases):
        d = load(Path(root) / name, cache)
        t0 = d['front'][0, 1]
        outs = core_on(d)
        ref = reference(d)
        ot = np.array([o.stamp for o in outs]) - t0
        ov = np.array([o.velocity for o in outs])
        rt = ref['vel'][:, 0] - t0
        sel = (rt > a) & (rt < b)
        ax.plot(rt[sel], ref['vel'][sel, 1], color=INK, lw=2.4, label='GNSS эталон')
        for key, color, label in (('front', BLUE, 'передняя тележка'), ('rear', ORANGE, 'задняя тележка')):
            w = d[key]
            m = (w[:, 1] - t0 > a) & (w[:, 1] - t0 < b)
            ax.plot(w[m, 1] - t0, w[m, 2] / 3.6, color=color, lw=1.2, marker='o', ms=2.5, label=label)
        m = (ot > a) & (ot < b)
        ax.plot(ot[m], ov[m], color=AQUA, lw=2.0, ls='--', label='оценка узла')
        ax.set_title(title, fontsize=11)
        ax.set_xlim(a, b)
        ax.set_xlabel('время от начала записи, с')
        ax.set_ylabel('скорость, м/с')
    axes[0, 0].legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    save(fig, 'slip_cases.png')


def plot_along(root, cache, name='30618_88548b02'):
    """Вдольпутевая ошибка как разность дуговых координат оценки и истины на карте.

    Истина: RTK фиксы master без скачков метки времени.
    """
    from build_map import project
    d = load(Path(root) / name, cache)
    doc = json.loads((ROOT / 'src/odometria/odometria/data/track_map.json').read_text(encoding='utf-8'))
    line = np.asarray(doc['points_enu'])
    junction = int(doc['east_junction_s'])
    fix = d['master_fix']
    origin = tuple(fix[fix[:, 5] >= 0][0, 2:5])
    shift = to_enu(*origin) - to_enu(*origin, origin=origin)
    offset = fix[:, 1] - fix[:, 0]
    rover = fix[(fix[:, 5] == 2) & (np.abs(offset - np.median(offset)) < 0.3)]
    truth = to_enu(rover[:, 2], rover[:, 3], rover[:, 4])
    start_west = truth[0, 0] < -2000
    part = line[:junction + 30] if start_west else line[junction - 30:]
    s_true, _, d_true = project(part, truth)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for use, color, label in ((False, ORANGE, 'без ориентиров: только колёса и карта'),
                              (True, BLUE, 'с ориентирами остановок')):
        core = OdometryCore(CoreConfig(use_landmarks=use), track_map=TrackMap.load(), model=TractionModel.load())
        outs, _ = run_core(d, core)
        ot = np.array([o.stamp for o in outs])
        op = np.array([o.position for o in outs])
        j, ok = match(rover[:, 1], ot)
        s_est, _, d_est = project(part, op[j[ok]] + shift)
        good = (d_est < 5) & (d_true[ok] < 3)
        along = s_est[good] - s_true[ok][good]
        ax.plot((s_true[ok][good] - s_true[ok][good][0]) / 1000, along, color=color, lw=1.8, label=label)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel('пройдено по карте, км')
    ax.set_ylabel('вдольпутевая ошибка, м')
    ax.set_title('Поездка с масштабом колёс +0,7 %: остановки гасят дрейф и оценивают масштаб')
    ax.legend(loc='lower left')
    save(fig, 'along_track.png')


def plot_bags(report_full, report_nolm):
    full = {b['bag']: b for b in json.loads(Path(report_full).read_text(encoding='utf-8'))['bags']}
    nolm = {b['bag']: b for b in json.loads(Path(report_nolm).read_text(encoding='utf-8'))['bags']}
    names = [n for n in full if full[n].get('out2ref.p_rmse_3d') is not None and nolm.get(n, {}).get('out2ref.p_rmse_3d')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, title, unit in ((axes[0], 'p_rmse_3d', 'СКО положения 3D по записям', 'м'),
                                 (axes[1], 'v_rmse_clean', 'СКО скорости по записям', 'м/с')):
        for src, color, label in ((nolm, ORANGE, 'без ориентиров'), (full, BLUE, 'итоговое решение')):
            v = np.sort([src[n][f'out2ref.{key}'] for n in names if src[n].get(f'out2ref.{key}') is not None])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=color, label=label)
        ax.set_xscale('log')
        ax.set_xlabel(f'{title.split(" по")[0]}, {unit} (логарифм)')
        ax.set_ylabel('доля записей')
        ax.set_title(title)
    axes[0].legend(loc='lower right')
    fig.tight_layout()
    save(fig, 'per_bag_cdf.png')


def plot_model():
    doc = json.loads((ROOT / 'src/odometria/odometria/data/traction_model.json').read_text(encoding='utf-8'))
    table = np.asarray(doc['table'])
    u = np.round(np.asarray(doc['u_knots']) * 15).astype(int)
    v = np.asarray(doc['v_knots'])
    cmap = LinearSegmentedColormap.from_list('div', ['#b42424', '#e34948', '#f0efec', '#3987e5', '#184f95'])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    mesh = ax.pcolormesh(np.arange(len(v) + 1) - 0.5, np.arange(len(u) + 1) - 0.5, table,
                         cmap=cmap, norm=TwoSlopeNorm(0.0, -1.6, 1.2), edgecolors=SURFACE, linewidth=0.6)
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels([f'{x:g}' for x in v])
    ax.set_yticks(range(0, len(u), 3))
    ax.set_yticklabels(u[::3])
    ax.set_xlabel('скорость, м/с')
    ax.set_ylabel('позиция контроллера')
    ax.grid(False)
    ax.set_title(f'Табличная модель привода T(u, v), м/с² (τ = {doc["tau_s"]} с, СКО {doc["holdout_rmse_mps2"]} м/с²)')
    bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label('ускорение без уклона, м/с²')
    bar.outline.set_visible(False)
    save(fig, 'traction_model.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/replay_dayout.json')
    parser.add_argument('--report-nolm', type=Path, default=ROOT / 'reports/replay_dayout_nolandmarks.json')
    args = parser.parse_args()
    plot_map()
    plot_model()
    plot_scale(args.data, args.cache)
    plot_slip(args.data, args.cache)
    plot_along(args.data, args.cache)
    if args.report.exists() and args.report_nolm.exists():
        plot_bags(args.report, args.report_nolm)


if __name__ == '__main__':
    main()
