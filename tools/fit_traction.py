"""Офлайн калибровка табличной модели тягового привода.

Модель продольного ускорения:
    a = T(u_e, v) - g * i(s) + b
u_e  отклик привода на позицию контроллера u/15 (апериодическое звено tau);
T    таблица тяги и торможения на узлах (u_e, v), билинейная интерполяция;
i(s) уклон пути из высот карты;
b    адаптивная поправка, оценивается онлайн фильтром.

Цель регрессии: ускорение по сглаженной скорости колёс на участках, где
тележки согласованы. Таблица регуляризована вторыми разностями.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix, lil_matrix, vstack
from scipy.sparse.linalg import lsqr

from bagdata import load, unique_bags
from build_map import project, rtk_trip

ROOT = Path(__file__).resolve().parents[1]
U_KNOTS = np.arange(-15, 16) / 15.0
V_KNOTS = np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17.0])
G = 9.80665


def lagged(u, tau, dt):
    k = 1.0 - np.exp(-dt / tau)
    out = np.empty(len(u))
    x = 0.0
    for i, ui in enumerate(u / 15.0):
        x += k * (ui - x)
        out[i] = x
    return out


def knots(x, grid):
    i = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
    w = np.clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0, 1)
    return i, w


def design(ue, v):
    iu, wu = knots(ue, U_KNOTS)
    iv, wv = knots(np.clip(v, 0, V_KNOTS[-1]), V_KNOTS)
    nv = len(V_KNOTS)
    rows = np.repeat(np.arange(len(ue)), 4)
    cols = np.stack([iu * nv + iv, (iu + 1) * nv + iv, iu * nv + iv + 1, (iu + 1) * nv + iv + 1], 1).ravel()
    vals = np.stack([(1 - wu) * (1 - wv), wu * (1 - wv), (1 - wu) * wv, wu * wv], 1).ravel()
    return csr_matrix((vals, (rows, cols)), shape=(len(ue), len(U_KNOTS) * nv))


def smoothness(weight):
    nu, nv = len(U_KNOTS), len(V_KNOTS)
    reg = lil_matrix(((nu - 2) * nv + nu * (nv - 2), nu * nv))
    k = 0
    for i in range(1, nu - 1):
        for j in range(nv):
            reg[k, (i - 1) * nv + j], reg[k, i * nv + j], reg[k, (i + 1) * nv + j] = weight, -2 * weight, weight
            k += 1
    for i in range(nu):
        for j in range(1, nv - 1):
            reg[k, i * nv + j - 1], reg[k, i * nv + j], reg[k, i * nv + j + 1] = weight, -2 * weight, weight
            k += 1
    return reg.tocsr()


def samples(data, circuit, grade, dt=0.05):
    f, r, c = data['front'], data['rear'], data['cmd']
    t0 = max(f[0, 1], r[0, 1], c[0, 1]) + 3
    t1 = min(f[-1, 1], r[-1, 1], c[-1, 1]) - 3
    if t1 - t0 < 60:
        return None
    t = np.arange(t0, t1, dt)
    fv = np.interp(t, f[:, 1], f[:, 2] / 3.6)
    rv = np.interp(t, r[:, 1], r[:, 2] / 3.6)
    u = c[np.clip(np.searchsorted(c[:, 1], t, side='right') - 1, 0, None), 2]
    fa = t - f[np.clip(np.searchsorted(f[:, 1], t, side='right') - 1, 0, None), 1]
    ra = t - r[np.clip(np.searchsorted(r[:, 1], t, side='right') - 1, 0, None), 1]
    v = 0.5 * (fv + rv)
    a = np.gradient(gaussian_filter1d(v, 3), t)
    bad = (np.abs(fv - rv) >= 0.12) | (fa > 0.3) | (ra > 0.3)
    clean = np.convolve(bad.astype(float), np.ones(41), 'same') == 0
    slope = np.full(len(t), np.nan)
    trip = rtk_trip(data)
    if trip is not None:
        s, _, dist = project(circuit, trip['E'])
        ok = dist < 2.0
        if ok.sum() > 100:
            si = np.interp(t, trip['t'][ok], s[ok])
            slope = grade[np.clip(np.round(si).astype(int), 0, len(grade) - 1)]
    return {'t': t, 'u': u, 'v': v, 'a': a, 'clean': clean, 'grade': slope}


def fit(sets, tau, weight=3.0):
    xs, ys = [], []
    for o in sets:
        m = o['clean'] & (o['v'] > 0.3) & np.isfinite(o['grade'])
        ue = lagged(o['u'], tau, 0.05)
        xs.append(design(ue[m], o['v'][m]))
        ys.append(o['a'][m] + G * o['grade'][m])
    x = vstack(xs)
    y = np.concatenate(ys)
    reg = smoothness(weight)
    theta = lsqr(vstack([x, reg]), np.r_[y, np.zeros(reg.shape[0])], atol=1e-9, btol=1e-9, iter_lim=10000)[0]
    return theta, x, y


def rmse(theta, sets, tau):
    errs = []
    for o in sets:
        m = o['clean'] & (o['v'] > 0.3) & np.isfinite(o['grade'])
        ue = lagged(o['u'], tau, 0.05)
        errs.append(design(ue[m], o['v'][m]) @ theta - o['a'][m] - G * o['grade'][m])
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--cache', type=Path, default=None)
    parser.add_argument('--map', type=Path, default=ROOT / 'src/odometria/odometria/data/track_map.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'src/odometria/odometria/data/traction_model.json')
    args = parser.parse_args()
    doc = json.loads(args.map.read_text(encoding='utf-8'))
    circuit = np.asarray(doc['points_enu'])
    grade = np.gradient(gaussian_filter1d(circuit[:, 2], 10, mode='nearest'))
    sets = []
    for folder in unique_bags(args.data, args.cache):
        o = samples(load(folder, args.cache), circuit, grade)
        if o is not None and np.isfinite(o['grade']).any():
            sets.append(o)
    train, test = sets[::2], sets[1::2]
    best = None
    for tau in (0.2, 0.3, 0.4, 0.5):
        theta, _, _ = fit(train, tau)
        score = rmse(theta, test, tau)
        print(f'tau {tau:.1f}: RMSE ускорения на отложенных {score:.4f} м/с²')
        if best is None or score < best[0]:
            best = (score, tau)
    tau = best[1]
    theta, _, _ = fit(sets, tau)
    table = theta.reshape(len(U_KNOTS), len(V_KNOTS))
    out = {
        'description': 'Табличная модель ускорения T(u_e, v), м/с², без уклона',
        'tau_s': tau,
        'u_knots': [round(float(x), 6) for x in U_KNOTS],
        'v_knots': [float(x) for x in V_KNOTS],
        'table': [[round(float(x), 4) for x in row] for row in table],
        'gravity': G,
        'holdout_rmse_mps2': round(best[0], 4),
        'bags': len(sets),
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('tau', tau, 'записано', args.output)


if __name__ == '__main__':
    main()
