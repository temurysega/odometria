"""Графики для README: python tools/plots.py DATA --cache CACHE [--check-bag BAG]"""
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
from build_map import project  # noqa: E402
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
    p = np.asarray(doc['points'])
    j = int(doc['east_junction_s'])
    tm = TrackMap.load()
    official = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((ROOT / 'official_maps').glob('*.json'))]
    fig = plt.figure(figsize=(12, 6.2))
    ax = fig.add_axes([0.06, 0.08, 0.61, 0.84])
    ax.plot(p[:j, 0], p[:j, 1], color=BLUE, lw=1.6, label='на восток')
    ax.plot(p[j:, 0], p[j:, 1], color=ORANGE, lw=1.6, label='на запад и кольца')
    for k, doc_o in enumerate(official):
        o = np.array([[q['x'], q['y']] for q in doc_o['points']])
        ax.plot(o[:, 0], o[:, 1], color=INK, lw=0.8, ls=(0, (4, 4)),
                label='карта организаторов' if k == 0 else None)
    for m in tm.stops:
        x, y, _ = tm.point(m['s'])
        ax.plot(x, y, 'o', ms=4 + 8 * m['share'], mfc=SURFACE, mec=INK, mew=1.2, zorder=3)
    ax.plot([], [], 'o', ms=8, mfc=SURFACE, mec=INK, mew=1.2, label='ориентир остановки (размер: частота)')
    ax.set_aspect('equal')
    ax.set_xlabel('x MGRS (восток), м')
    ax.set_ylabel('y MGRS (север), м')
    for k, spur in enumerate(tm.spurs):
        q = np.asarray(spur.line.points)
        ax.plot(q[:, 0], q[:, 1], color=AQUA, lw=1.6, label='тупики конечной' if k == 0 else None)
    ax.set_title(f'Карта пути в MGRS: контур {tm.length / 1000:.2f} км, {len(tm.stops)} ориентиров, {len(tm.spurs)} тупика')
    ax.legend(loc='upper left', fontsize=9)
    s_all = np.arange(len(p))
    boxes = (((s_all > j - 150) & (s_all < j + 260), 'Восточное кольцо'),
             ((s_all > len(p) - 330) | (s_all < 120), 'Западное кольцо'))
    for k, (sel, title) in enumerate(boxes):
        x0, x1 = p[sel, 0].min() - 20, p[sel, 0].max() + 20
        y0, y1 = p[sel, 1].min() - 20, p[sel, 1].max() + 20
        sub = fig.add_axes([0.71, 0.55 - 0.47 * k, 0.27, 0.38])
        sub.plot(p[:j, 0], p[:j, 1], color=BLUE, lw=1.6)
        sub.plot(p[j:, 0], p[j:, 1], color=ORANGE, lw=1.6)
        for spur in tm.spurs:
            q = np.asarray(spur.line.points)
            sub.plot(q[:, 0], q[:, 1], color=AQUA, lw=1.6)
        sub.set_xlim(x0, x1)
        sub.set_ylim(y0, y1)
        sub.set_aspect('equal', adjustable='datalim')
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
    """Вдольпутевая ошибка base_link против эталона по обеим антеннам."""
    d = load(Path(root) / name, cache)
    doc = json.loads((ROOT / 'src/odometria/odometria/data/track_map.json').read_text(encoding='utf-8'))
    line = np.asarray(doc['points'])
    junction = int(doc['east_junction_s'])
    ref = reference(d)
    rp = ref['pos'][ref['pos'][:, 4] > 0]
    start_west = rp[0, 1] < 101000
    part = line[:junction + 30] if start_west else line[junction - 30:]
    s_true, _, d_true = project(part, rp[:, 1:4])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for use, color, label in ((False, ORANGE, 'без ориентиров: только колёса и карта'),
                              (True, BLUE, 'с ориентирами остановок')):
        core = OdometryCore(CoreConfig(use_landmarks=use), track_map=TrackMap.load(), model=TractionModel.load())
        outs, _ = run_core(d, core)
        outs = [o for o in outs if o.position_valid]
        ot = np.array([o.stamp for o in outs])
        op = np.array([o.position for o in outs])
        j, ok = match(rp[:, 0], ot)
        s_est, _, d_est = project(part, op[j[ok]])
        good = (d_est < 5) & (d_true[ok] < 3)
        along = s_est[good] - s_true[ok][good]
        ax.plot((s_true[ok][good] - s_true[ok][good][0]) / 1000, along, color=color, lw=1.8, label=label)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel('пройдено по карте, км')
    ax.set_ylabel('вдольпутевая ошибка base_link, м')
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


def plot_judge(bag):
    """Проверочный bag организаторов: ошибка по времени и западная конечная."""
    import judge_check
    data = judge_check.bagdata.read_bag(bag)
    ref = data['reference']
    runs = {}
    for flag in (False, True):
        core = OdometryCore(CoreConfig(gnss_correction=flag), track_map=TrackMap.load(), model=TractionModel.load())
        rows = judge_check.run(data, core)
        pairs = judge_check.synchronize(
            [(r[0], 0, r[1], (r[1],) + tuple(r[2:5])) for r in ref]
            + [(rec, 1, o.stamp, tuple(o.position)) for rec, o in rows if o.position_valid])
        runs[flag] = (np.array([(a[0], np.linalg.norm(np.subtract(b, a[1:]))) for a, b in pairs]),
                      np.array([(o.stamp,) + tuple(o.position) for _, o in rows if o.position_valid]))
    t0 = ref[0][1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]
    for flag, color, label in ((False, ORANGE, 'GNSS только для выставки'), (True, BLUE, 'итог: и пачки GNSS по ходу')):
        err = runs[flag][0]
        rmse = np.sqrt(np.mean(err[:, 1] ** 2))
        ax.plot(err[:, 0] - t0, np.maximum(err[:, 1], 0.03), color=color, lw=1.2, label=f'{label}, RMSE {rmse:.2f} м')
    fixes = np.concatenate([np.asarray(data[k])[:, 1] for k in ('master_fix', 'rover_fix') if k in data])
    ax.plot(fixes - t0, np.full(len(fixes), 0.022), '|', color=INK2, ms=7, label='фиксы GNSS в bag')
    ax.set_yscale('log')
    ax.set_ylim(0.015, 100.0)
    ax.set_xlabel('время записи, с')
    ax.set_ylabel('ошибка положения 3D, м')
    ax.set_title('Проверочный bag 30618_88aea4d9, метрика судьи')
    ax.legend(loc='upper left', fontsize=9)
    ax = axes[1]
    tm = TrackMap.load()
    r = np.asarray(ref)
    box = r[:, 1] - t0 > 1215
    x0, x1 = r[box, 2].min() - 25, r[box, 2].max() + 25
    y0, y1 = r[box, 3].min() - 25, r[box, 3].max() + 25
    c = np.asarray(tm.points)
    ax.plot(c[:, 0], c[:, 1], color=GRID, lw=5, solid_capstyle='round', label='контур карты')
    for k, spur in enumerate(tm.spurs):
        q = np.asarray(spur.line.points)
        ax.plot(q[:, 0], q[:, 1], color=AQUA, lw=5, alpha=0.45, label='тупики из обучающих рейсов' if k == 0 else None)
    ax.plot(r[box, 2], r[box, 3], color=INK, lw=1.2, label='эталон судьи')
    for flag, color, label in ((False, ORANGE, 'выход без пачек'), (True, BLUE, 'выход итог')):
        out = runs[flag][1]
        sel = out[:, 0] - t0 > 1215
        ax.plot(out[sel, 1], out[sel, 2], color=color, lw=1.2, ls=(0, (4, 2)), label=label)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Последние 95 с: съезд в тупик')
    ax.set_xlabel('x MGRS, м')
    ax.legend(loc='lower right', fontsize=8.5)
    fig.tight_layout()
    save(fig, 'judge_check.png')


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
    parser.add_argument('--check-bag', type=Path, help='проверочный bag организаторов с эталоном судьи')
    parser.add_argument('--only', nargs='*', help='только эти графики: map, model, scale, slip, along, bags, judge')
    args = parser.parse_args()
    if args.only:
        for name in args.only:
            if name == 'map':
                plot_map()
            elif name == 'judge':
                plot_judge(args.check_bag)
            elif name == 'model':
                plot_model()
        return
    plot_map()
    plot_model()
    plot_scale(args.data, args.cache)
    plot_slip(args.data, args.cache)
    plot_along(args.data, args.cache)
    if args.report.exists() and args.report_nolm.exists():
        plot_bags(args.report, args.report_nolm)
    if args.check_bag:
        plot_judge(args.check_bag)


if __name__ == '__main__':
    main()
