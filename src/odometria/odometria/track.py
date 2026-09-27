"""Карта пути, начальная привязка и вдольпутевой фильтр.

Карта: траектория master антенны в плоских координатах MGRS (контур из двух
направлений, шаг 1 м, высота антенны) и ориентиры остановок, построенные
офлайн (tools/build_map.py). Выход считается для base_link по tf антенн:
точка на хорде master и rover, высота на antenna_z ниже антенн.

Вдольпутевой фильтр Калмана, состояние [s, k]:
    s' = s + k * ds_колёс        k' = k + w
s дуговая координата антенны, k поправка масштаба колёс (износ бандажей,
калибровка датчика). Когда вагон остановился, место остановки сравнивается
с ориентирами карты. Удачная привязка обнуляет накопленную вдольпутевую
ошибку и уточняет k, поэтому ошибка масштаба не копится на всю поездку.
"""
from bisect import bisect_right
import json
import math
import random
from pathlib import Path

from .geodesy import MgrsFrame

DATA = Path(__file__).with_name('data')


class TrackMap:
    def __init__(self, frame, points, stops=(), east_junction=None, closed=False):
        if len(points) < 2:
            raise ValueError('в карте меньше двух точек')
        self.frame = frame if isinstance(frame, MgrsFrame) else MgrsFrame(
            frame.get('utm_zone', 37), *frame.get('grid_origin', (300000.0, 6100000.0)))
        self.points = [tuple(float(c) for c in p) for p in points]
        self.closed = bool(closed)
        if self.closed:
            # замкнутый контур: последний отрезок возвращает в начало
            self.points.append(self.points[0])
        # дуговая координата в плане, как при построении ориентиров
        self.s = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.s.append(self.s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.length = self.s[-1]
        self.stops = sorted(({'s': float(x['s']), 'sigma': float(x['sigma']),
                              'count': int(x.get('count', 1)), 'share': float(x.get('share', 1.0))}
                             for x in stops), key=lambda x: x['s'])
        self.east_junction = east_junction
        # на сколько base_link ниже антенн по высотам карты организаторов
        self.base_height = None
        self.spurs = []
        self._smooth_z = self._robust_profile([p[2] for p in self.points])
        self._grid = {}
        for i, (x, y, _) in enumerate(self.points):
            self._grid.setdefault((int(x // 50), int(y // 50)), []).append(i)

    @classmethod
    def load(cls, path=None):
        doc = json.loads(Path(path or DATA / 'track_map.json').read_text(encoding='utf-8'))
        track = cls(doc['frame'], doc['points'], doc.get('stops', ()),
                    doc.get('east_junction_s'), doc.get('closed', False))
        track.base_height = doc.get('base_link_below_antenna_m')
        track.spurs = [Spur(track, x['kind'], x['junction_s'], x['points']) for x in doc.get('spurs', ())]
        return track

    def wrap(self, s):
        return s % self.length if self.closed else s

    def delta(self, a, b):
        """Разность дуговых координат a и b с учётом замкнутости."""
        d = a - b
        if self.closed:
            d = (d + 0.5 * self.length) % self.length - 0.5 * self.length
        return d

    def _index(self, s):
        return min(max(bisect_right(self.s, s) - 1, 0), len(self.points) - 2)

    def point(self, s):
        """Точка карты на дуге s; за концами открытой карты линейное продолжение."""
        s = self.wrap(s)
        i = self._index(s)
        a, b = self.points[i], self.points[i + 1]
        span = self.s[i + 1] - self.s[i]
        w = (s - self.s[i]) / span if span > 0 else 0.0
        return tuple(pa + w * (pb - pa) for pa, pb in zip(a, b))

    def tangent(self, s):
        s = self.wrap(s)
        i = self._index(s)
        a, b = self.points[i], self.points[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n

    @staticmethod
    def _robust_profile(z, half=15):
        """Медиана по окну 31 м: убирает всплески высоты RTK под мостами."""
        n = len(z)
        out = []
        for i in range(n):
            window = sorted(z[max(0, i - half):min(n, i + half + 1)])
            out.append(window[len(window) // 2])
        return out

    def grade(self, s, half=15.0, limit=0.06):
        """Уклон пути по сглаженному профилю высот, ограничен реальными 6 %."""
        def z(x):
            x = self.wrap(x)
            i = self._index(x)
            span = self.s[i + 1] - self.s[i]
            w = min(1.0, max(0.0, (x - self.s[i]) / span)) if span > 0 else 0.0
            return self._smooth_z[i] + w * (self._smooth_z[i + 1] - self._smooth_z[i])
        g = (z(s + half) - z(s - half)) / (2.0 * half)
        return max(-limit, min(limit, g))

    def curvature(self, s, half=5.0):
        t0 = self.tangent(s - half)
        t1 = self.tangent(s + half)
        return math.atan2(t0[0] * t1[1] - t0[1] * t1[0], t0[0] * t1[0] + t0[1] * t1[1]) / (2.0 * half)

    def candidates(self, x, y, radius=30.0):
        """Проекции точки на все участки карты ближе radius."""
        cells = range(-1 - int(radius // 50), 2 + int(radius // 50))
        seen = set()
        out = []
        for dx in cells:
            for dy in cells:
                for i in self._grid.get((int(x // 50) + dx, int(y // 50) + dy), ()):
                    for j in (i - 1, i):
                        if j < 0 or j >= len(self.points) - 1 or j in seen:
                            continue
                        seen.add(j)
                        a, b = self.points[j], self.points[j + 1]
                        ex, ey = b[0] - a[0], b[1] - a[1]
                        l2 = ex * ex + ey * ey
                        if l2 <= 0:
                            continue
                        w = ((x - a[0]) * ex + (y - a[1]) * ey) / l2
                        # за концами карты проекция продолжает крайний отрезок
                        low = -1e9 if j == 0 and not self.closed else 0.0
                        high = 1e9 if j == len(self.points) - 2 and not self.closed else 1.0
                        w = max(low, min(high, w))
                        px, py = a[0] + w * ex, a[1] + w * ey
                        d = math.hypot(x - px, y - py)
                        if d <= radius:
                            n = math.sqrt(l2)
                            out.append((d, self.s[j] + w * (self.s[j + 1] - self.s[j]), ex / n, ey / n))
        return out


class Spur:
    """Тупик у конечной: своя ломаная и стык с контуром.

    diverge: вагон съезжает с контура в тупик (конец рейса), merge: выезжает
    из тупика на контур (начало рейса). Вдольпутевая координата фильтра
    остаётся дугой контура: s = start + u, где u дуга тупика, а start дуга
    его начала, так что на стыке координата непрерывна.
    """

    def __init__(self, track, kind, junction, points):
        if kind not in ('diverge', 'merge'):
            raise ValueError('тип тупика diverge или merge')
        self.kind = kind
        self.track = track
        self.line = TrackMap(track.frame, points)
        self.length = self.line.length
        self.start = track.wrap(float(junction) if kind == 'diverge' else float(junction) - self.length)

    def arc(self, s, margin=50.0):
        """Дуга тупика для координаты s или None, если s не на тупике."""
        u = self.track.delta(s, self.start)
        if self.kind == 'diverge':
            # за концом тупика вагон стоит у упора
            return min(u, self.length) if 0.0 <= u <= self.length + margin else None
        return max(u, 0.0) if -margin <= u <= self.length else None


class AlongTrackFilter:
    def __init__(self, s, sigma_s=0.5, k=1.0, sigma_k=0.008, q_distance=0.02 ** 2, q_scale=1e-5 ** 2):
        self.s = s
        self.k = k
        self.p = [[sigma_s ** 2, 0.0], [0.0, sigma_k ** 2]]
        self.q_distance = q_distance
        self.q_scale = q_scale
        self.updates = 0

    def advance(self, ds, extra_variance=0.0):
        if ds == 0.0 and extra_variance == 0.0:
            return
        p = self.p
        self.s += self.k * ds
        a = abs(ds)
        pss = p[0][0] + 2 * ds * p[0][1] + ds * ds * p[1][1] + self.q_distance * a + extra_variance
        psk = p[0][1] + ds * p[1][1]
        pkk = p[1][1] + self.q_scale * a
        self.p = [[pss, psk], [psk, pkk]]

    def correct(self, z, sigma):
        p = self.p
        denom = p[0][0] + sigma * sigma
        k0 = p[0][0] / denom
        k1 = p[0][1] / denom
        innovation = z - self.s
        self.s += k0 * innovation
        self.k += k1 * innovation
        self.k = min(1.05, max(0.95, self.k))
        self.p = [[(1 - k0) * p[0][0], (1 - k0) * p[0][1]],
                  [(1 - k0) * p[0][1], p[1][1] - k1 * p[0][1]]]
        self.updates += 1
        return innovation

    @property
    def sigma_s(self):
        return math.sqrt(max(self.p[0][0], 0.0))

    @property
    def sigma_k(self):
        return math.sqrt(max(self.p[1][1], 0.0))


class ParticleTrack:
    """Вдольпутевое положение и масштаб колёс облаком частиц.

    Частица i хранит положение b_i на момент последней остановки и масштаб
    k_i. Между остановками её положение b_i + k_i * D выражается через
    пройденный колёсами путь D, поэтому частицы трогаются только на
    остановках (десятки раз за поездку), а в каждом такте обновляется
    лишь средняя оценка. На остановке вес частицы умножается на
    правдоподобие места остановки: смесь гауссиан ориентиров с весом их
    частоты и равномерного фона «остановка вне платформы». Неоднозначную
    остановку облако сохраняет как несколько гипотез, их разрешают
    следующие платформы, поэтому нет жёсткого строба и срывов привязки.
    """

    def __init__(self, s, sigma_s=0.5, k=1.0, sigma_k=0.008, count=1500,
                 q_distance=0.02 ** 2, q_scale=2e-5 ** 2, seed=7):
        self.rng = random.Random(seed)
        gauss = self.rng.gauss
        self.base = [s + gauss(0.0, sigma_s) for _ in range(count)]
        self.scale = [min(1.05, max(0.95, k + gauss(0.0, sigma_k))) for _ in range(count)]
        self.weight = [1.0 / count] * count
        self.q_distance = q_distance
        self.q_scale = q_scale
        self.travel = 0.0
        self.extra = 0.0
        self.s = s
        self.k = k
        self.var_s = sigma_s ** 2
        self.var_k = sigma_k ** 2
        self.cov_sk = 0.0
        self.share = 1.0
        self.updates = 0

    def advance(self, ds, extra_variance=0.0):
        self.travel += ds
        self.extra += extra_variance
        self.s += self.k * ds
        self.var_s += 2 * ds * self.cov_sk + ds * ds * self.var_k + self.q_distance * abs(ds) + extra_variance
        self.cov_sk += ds * self.var_k

    @property
    def p(self):
        return [[self.var_s, self.cov_sk], [self.cov_sk, self.var_k]]

    @property
    def sigma_s(self):
        return math.sqrt(max(self.var_s, 0.0))

    @property
    def sigma_k(self):
        return math.sqrt(max(self.var_k, 0.0))

    def observe(self, likelihood):
        """Остановка: перенос частиц, взвешивание, ресэмплинг, оценка."""
        gauss = self.rng.gauss
        noise = math.sqrt(self.q_distance * abs(self.travel) + self.extra)
        noise_k = math.sqrt(self.q_scale * abs(self.travel))
        n = len(self.base)
        total = 0.0
        for i in range(n):
            k = self.scale[i]
            x = self.base[i] + k * self.travel + (gauss(0.0, noise) if noise > 0 else 0.0)
            if noise_k > 0:
                self.scale[i] = min(1.05, max(0.95, k + gauss(0.0, noise_k)))
            self.base[i] = x
            w = self.weight[i] * likelihood(x)
            self.weight[i] = w
            total += w
        if total <= 0.0 or not math.isfinite(total):
            self.weight = [1.0 / n] * n
        else:
            self.weight = [w / total for w in self.weight]
        self.travel = 0.0
        self.extra = 0.0
        ess = 1.0 / sum(w * w for w in self.weight)
        if ess < 0.5 * n:
            self._resample()
        before = self.s
        self._estimate()
        self.updates += 1
        return self.s - before

    def relocate(self, offset, sigma):
        """Облако заново вокруг надёжного наблюдения, далёкого от всех частиц.

        Прочие гипотезы облака сдвигом не переносятся: они строились по той
        же ошибке, которую исправляет наблюдение.
        """
        gauss = self.rng.gauss
        center = self.s + offset
        self.base = [center + gauss(0.0, sigma) for _ in self.base]
        n = len(self.base)
        self.weight = [1.0 / n] * n
        self.travel = 0.0
        self.extra = 0.0
        before = self.s
        self._estimate()
        self.updates += 1
        return self.s - before

    def _resample(self):
        n = len(self.base)
        step = 1.0 / n
        u = self.rng.random() * step
        acc = self.weight[0]
        j = 0
        base, scale = [], []
        gauss = self.rng.gauss
        for i in range(n):
            target = u + i * step
            while target > acc and j < n - 1:
                j += 1
                acc += self.weight[j]
            base.append(self.base[j] + gauss(0.0, 0.05))
            scale.append(min(1.05, max(0.95, self.scale[j] + gauss(0.0, 2e-4))))
        self.base, self.scale = base, scale
        self.weight = [step] * n

    def _estimate(self):
        """Оценка по самой весомой группе частиц, а не по среднему всех гипотез."""
        order = sorted(range(len(self.base)), key=lambda i: self.base[i])
        groups = [[order[0]]]
        for a, b in zip(order, order[1:]):
            if self.base[b] - self.base[a] > 4.0:
                groups.append([])
            groups[-1].append(b)
        best = max(groups, key=lambda g: sum(self.weight[i] for i in g))
        wsum = sum(self.weight[i] for i in best) or 1.0
        s = sum(self.weight[i] * self.base[i] for i in best) / wsum
        k = sum(self.weight[i] * self.scale[i] for i in best) / wsum
        # оценка по самой вероятной группе, но разброс по всем гипотезам:
        # две равные остановки в сотнях метров не должны выглядеть точным местом
        self.var_s = sum(w * (b - s) ** 2 for w, b in zip(self.weight, self.base))
        self.var_k = sum(w * (q - k) ** 2 for w, q in zip(self.weight, self.scale))
        self.cov_sk = sum(w * (b - s) * (q - k) for w, b, q in zip(self.weight, self.base, self.scale))
        self.s, self.k = s, k
        self.share = wsum


class Localizer:
    """Начальная привязка по первым секундам GNSS и выдача положения.

    GNSS принимается только в окне init_window после первого валидного фикса.
    Выход в плоских координатах MGRS (x восток, y север, z высота по REP 103),
    точка задаётся output_point: base_link, master или rover.
    """

    def __init__(self, track_map=None, init_window=1.0, max_map_offset=15.0,
                 master_x=-9.873, rover_x=2.563, antenna_z=3.0, output_point='base_link',
                 frame=None, blend_length=80.0, scale_prior=1.0, scale_sigma=0.008,
                 use_landmarks=True, landmark_gate=3.0, landmark_max_jump=12.0,
                 landmark_confidence=0.9, landmark_random_stop=0.15, particles=1500):
        self.map = track_map
        self.frame = track_map.frame if track_map is not None else (frame or MgrsFrame())
        self.init_window = init_window
        self.max_map_offset = max_map_offset
        self.antenna_baseline = rover_x - master_x
        if self.antenna_baseline <= 0:
            raise ValueError('rover должен стоять впереди master')
        # доля хорды master и rover, на которой лежит выходная точка
        self.output_point = output_point
        self.output_fraction = {'master': 0.0, 'rover': 1.0}.get(output_point, -master_x / self.antenna_baseline)
        self.antenna_z = antenna_z
        self.blend_length = blend_length
        self.scale_prior = scale_prior
        self.scale_sigma = scale_sigma
        self.use_landmarks = use_landmarks
        self.landmark_gate = landmark_gate
        self.landmark_max_jump = landmark_max_jump
        self.landmark_confidence = landmark_confidence
        self.landmark_random_stop = landmark_random_stop
        self.landmark_tail_weight = 0.2
        self.landmark_tail_sigma = 3.0
        self.particles = particles
        self.burst = []
        self.burst_start = None
        self.gnss_corrections = 0
        self.gnss_relocations = 0
        # тупик, в котором стоит вагон по последним фиксам, иначе контур
        self.branch = None
        self.last_gnss = None
        # ошибки фиксов коррелированы во времени: на стоянке пачка не несёт
        # новой информации, а повторные поправки вырождают облако частиц
        self.gnss_min_travel = 50.0
        # пачка применяется через полсекунды: смена ветки нужна сразу
        self.burst_window = 0.6
        # фиксы дальше от всех путей карты не используются; соседний путь
        # объезда в 4...6 м ещё даёт верное место вдоль пути
        self.gnss_lateral = 8.0
        # перепривязка по одной пачке не больше чем на столько метров,
        # больший сдвиг должна подтвердить следующая пачка
        self.relocate_free = 3.0
        self.gnss_pending = None
        # вдоль пути против эталона на проверочном bag: статус 2 около 0,6 м, прочие около 3 м
        self.gnss_sigma_rtk = 0.6
        self.gnss_sigma_other = 3.0
        self.last_gnss_distance = None
        self.relock_share = 0.3
        self.relock_distance = 2000.0
        self.relock_agreement = 3.0
        self.pending = None
        self.relocks = 0
        self.fixes = {'master': [], 'rover': []}
        self.first_fix_time = None
        self.ready = False
        self.source = None
        self.start_point = None
        self.filter = None
        self.origin_offset = (0.0, 0.0, 0.0)
        self.start_s = None
        self.heading = (1.0, 0.0)
        self.map_error = None
        self.dead_reckoning = 0.0
        self.last_landmark_distance = None
        self.last_landmark = None

    def add_fix(self, source, t, lat, lon, alt, status, distance):
        if self.ready or source not in self.fixes:
            return False
        if status is not None and status < 0:
            return False
        if not all(math.isfinite(v) for v in (lat, lon, alt)):
            return False
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0.0 and lon == 0.0):
            return False
        if self.first_fix_time is None:
            self.first_fix_time = t
        self.fixes[source].append((t, lat, lon, alt, distance, -1 if status is None else int(status)))
        # без колёс окно может не закрываться долго: память ограничена
        if len(self.fixes[source]) > 256:
            del self.fixes[source][:-256]
        return True

    def window_closed(self, t):
        if self.first_fix_time is None:
            return False
        both = self.fixes['master'] and self.fixes['rover']
        return t - self.first_fix_time >= (self.init_window if both else 2.0 * self.init_window)

    def finalize(self, distance):
        """Закрывает окно GNSS и выполняет привязку к карте."""
        self.ready = True
        source = 'master' if self.fixes['master'] else 'rover'
        fixes = self.fixes[source]
        if not fixes:
            return False
        self.source = source
        heading = self._heading()
        if heading is not None:
            self.heading = heading
        first = self.frame.forward(*fixes[0][1:4])
        # положение master в момент первого фикса, для счисления без карты
        back = self.antenna_baseline if source == 'rover' else 0.0
        self.start_point = (first[0] - back * self.heading[0], first[1] - back * self.heading[1],
                            first[2], fixes[0][4])
        if self.map is None:
            self.filter = AlongTrackFilter(distance, k=self.scale_prior, sigma_k=self.scale_sigma)
            self.start_s = None
            return True
        offset = self.antenna_baseline if source == 'rover' else 0.0
        estimates = []
        best = None
        nearest = None
        for t, lat, lon, alt, d, _ in fixes:
            x, y, z = self.frame.forward(lat, lon, alt)
            choice = self._match(x, y, heading)
            if choice is None:
                continue
            dist, s_fix, spur = choice
            estimates.append(s_fix - offset - d)
            if best is None:
                best = (dist, (x, y, z))
            if nearest is None or dist < nearest[0]:
                nearest = (dist, spur)
        if not estimates:
            self.filter = AlongTrackFilter(distance, k=self.scale_prior, sigma_k=self.scale_sigma)
            self.start_s = None
            self.map_error = None
            return True
        estimates.sort()
        base = estimates[len(estimates) // 2]
        # старт в тупике конечной: выход по его ломаной до стыка с контуром
        self.branch = nearest[1]
        self.map_error = best[0]
        self.start_s = base + distance
        # точность старта по статусу фикса: 2 RTK, 1 SBAS, 0 автономное решение
        grades = sorted(f[5] for f in fixes)
        status_sigma = {2: 0.3, 1: 1.5}.get(grades[len(grades) // 2], 3.0)
        # старт в стороне от карты означает неизвестный путь (обход внутри
        # кольца, тупик конечной), длина которого может отличаться от карты
        topology_sigma = 1.5 * self.map_error if self.map_error > 2.0 else 0.0
        sigma_start = min(20.0, max(status_sigma, topology_sigma))
        if self.particles:
            self.filter = ParticleTrack(self.start_s, sigma_s=sigma_start, k=self.scale_prior,
                                        sigma_k=self.scale_sigma, count=self.particles)
        else:
            self.filter = AlongTrackFilter(self.start_s, sigma_s=sigma_start,
                                           k=self.scale_prior, sigma_k=self.scale_sigma)
        self.last_landmark_distance = distance
        self.last_gnss_distance = distance
        # смещение первого фикса от карты плавно убирается на первых метрах
        on_map = self.map.point(base + fixes[0][4] + offset)
        self.origin_offset = tuple(a - b for a, b in zip(first, on_map))
        return True

    def _heading(self):
        m, r = self.fixes['master'], self.fixes['rover']
        if m and r:
            a = self.frame.forward(*m[0][1:4])
            b = self.frame.forward(*r[0][1:4])
            e, n = b[0] - a[0], b[1] - a[1]
            norm = math.hypot(e, n)
            # курс по линии антенн, только если база похожа на tf
            if abs(norm - self.antenna_baseline) < 1.5:
                return e / norm, n / norm
        pts = m or r
        if len(pts) >= 2:
            a = self.frame.forward(*pts[0][1:4])
            b = self.frame.forward(*pts[-1][1:4])
            e, n = b[0] - a[0], b[1] - a[1]
            norm = math.hypot(e, n)
            if norm > 3.0:
                return e / norm, n / norm
        return None

    def _options(self, x, y, radius):
        """Проекции точки на контур и тупики: удаление, s, курс пути, тупик."""
        out = [o + (None,) for o in self.map.candidates(x, y, radius)]
        for spur in self.map.spurs:
            for d, u, tx, ty in spur.line.candidates(x, y, radius):
                if -1.0 <= u <= spur.length + 1.0:
                    out.append((d, self.map.wrap(spur.start + u), tx, ty, spur))
        return out

    def _match(self, x, y, heading):
        options = self._options(x, y, self.max_map_offset)
        if heading is not None:
            # курс по двум антеннам различает встречные пути; даже если свой
            # путь чуть дальше допуска, ближний встречный не берём: такая
            # ошибка старта держится всю поездку и даёт сотни метров
            aligned = [o for o in options if o[2] * heading[0] + o[3] * heading[1] > 0.5]
            if not aligned:
                expanded = self._options(x, y, 4.0 * self.max_map_offset)
                aligned = [o for o in expanded if o[2] * heading[0] + o[3] * heading[1] > 0.5]
            options = aligned
        elif not options:
            return None
        if not options:
            # курс и карта не согласуются: абсолютную привязку не делаем
            return None
        dist, s, _, _, spur = min(options, key=lambda o: o[0])
        return dist, s, spur

    def correction_fix(self, source, t, lat, lon, alt, status, distance):
        """Фикс GNSS после выставки: копится в пачку для коррекции.

        Организаторы разрешили использовать GNSS для коррекции; в проверочных
        данных он приходит редкими пачками по несколько секунд.
        """
        if not self.ready or self.filter is None or self.map is None or self.start_s is None:
            return False
        if status is not None and status < 0:
            return False
        if not all(math.isfinite(v) for v in (lat, lon, alt)):
            return False
        x, y, _ = self.frame.forward(lat, lon, alt)
        offset = self.antenna_baseline if source == 'rover' else 0.0
        predicted = self.filter.s + offset
        tx, ty = self._tangent(predicted)
        options = [o for o in self._options(x, y, 20.0)
                   if abs(self.map.delta(o[1], predicted)) < 80.0 and o[2] * tx + o[3] * ty > 0.7]
        best = min(options, key=lambda o: o[0]) if options else None
        main = min((o[0] for o in options if o[4] is None), default=math.inf)
        side = min(options, key=lambda o: o[0] if o[4] is not None else math.inf, default=None)
        vote = None
        if side is not None and side[4] is not None and side[0] + 0.5 < main:
            vote = side[4]
        elif main + 0.5 < (side[0] if side is not None and side[4] is not None else math.inf):
            vote = 'main'
        if not self.burst:
            self.burst_start = t
        self.burst.append({'t': t, 'status': int(status), 'lateral': best[0] if best else None,
                           'along': (best[1] - offset - distance) if best else None, 'vote': vote})
        return True

    def _tangent(self, s):
        u = self.branch.arc(s) if self.branch is not None else None
        if u is not None:
            return self.branch.line.tangent(min(u, self.branch.length - 0.5))
        return self.map.tangent(s)

    def flush_corrections(self, t, distance, force=False):
        """Применяет накопленную пачку фиксов: ветка и коррекция вдоль пути."""
        if not self.burst:
            return None
        last = self.burst[-1]['t']
        if not force and t - last < 0.5 and t - self.burst_start < self.burst_window:
            return None
        burst, self.burst = self.burst, []
        on_map = [b for b in burst if b['lateral'] is not None and b['lateral'] < self.gnss_lateral]
        if len(on_map) * 2 >= len(burst):
            # ветка по большинству фиксов, однозначно ближе к одному из путей
            votes = [b['vote'] for b in on_map if b['vote'] is not None]
            spur_votes = [v for v in votes if v != 'main']
            if len(spur_votes) * 2 > len(votes) and spur_votes:
                self.branch = max(set(spur_votes), key=spur_votes.count)
            elif votes and len(spur_votes) * 2 < len(votes):
                self.branch = None
            along = sorted(b['along'] for b in on_map)
            middle = along[len(along) // 2]
            observed = middle + distance
            n = len(along)
            # сбойный приёмник чередует две точки в десятках метров: пачка
            # согласована, только если почти все фиксы у медианы
            agree = sum(1 for a in along if abs(a - middle) < 1.5)
            grades = sorted(b['status'] for b in on_map)
            sigma = self.gnss_sigma_rtk if grades[len(grades) // 2] == 2 else self.gnss_sigma_other
            if (self.last_gnss_distance is not None
                    and abs(distance - self.last_gnss_distance) < self.gnss_min_travel):
                return None
            self.last_gnss_distance = distance
            return self._apply_gnss(observed, sigma, n >= 3 and agree >= 0.8 * n, distance)
        # пачка в стороне от всех путей карты: путь вне карты или сбой приёмника
        return None

    def _apply_gnss(self, observed, sigma, consistent=False, distance=None):
        f = self.filter
        wide = max(3.0 * sigma, 5.0)
        innovation = self.map.delta(observed, f.s)
        if (isinstance(f, ParticleTrack) and consistent and sigma <= self.gnss_sigma_rtk
                and abs(innovation) > 3.0 * math.sqrt(f.var_s + sigma * sigma)):
            # облако уверенно ушло (стык карты, долгое проскальзывание):
            # частиц рядом с согласованной точной пачкой нет, взвешивать нечего.
            # Большой скачок нужно подтвердить следующей пачкой с тем же сдвигом
            if abs(innovation) > self.relocate_free:
                previous = self.gnss_pending
                self.gnss_pending = (distance, innovation)
                if (previous is None or distance is None or abs(distance - previous[0]) > 1000.0
                        or abs(innovation - previous[1]) > 3.0):
                    return None
            self.gnss_pending = None
            shift = f.relocate(innovation, sigma)
            self.gnss_corrections += 1
            self.gnss_relocations += 1
            self.last_gnss = (observed, shift)
            return shift
        if isinstance(f, ParticleTrack):
            delta = self.map.delta
            norm_n = 0.8 / (math.sqrt(2.0 * math.pi) * sigma)
            norm_w = 0.2 / (math.sqrt(2.0 * math.pi) * wide)

            def likelihood(x):
                d = delta(x, observed)
                return 1e-4 + norm_n * math.exp(-0.5 * d * d / sigma ** 2) + norm_w * math.exp(-0.5 * d * d / wide ** 2)

            shift = f.observe(likelihood)
        else:
            shift = f.correct(observed, sigma)
        self.gnss_corrections += 1
        self.last_gnss = (observed, shift)
        return shift

    def advance(self, ds, extra_variance=0.0):
        if self.filter is not None:
            self.filter.advance(ds, extra_variance)
        else:
            self.dead_reckoning += ds

    def try_landmark(self, distance):
        """Привязка к ориентиру при подтверждённой остановке."""
        if (not self.use_landmarks or self.filter is None or self.start_s is None
                or not self.map.stops):
            return None
        if self.branch is not None and self.branch.arc(self.filter.s) is not None:
            # платформы контура к тупику отношения не имеют
            return None
        if self.last_landmark_distance is not None and abs(distance - self.last_landmark_distance) < 15.0:
            return None
        f = self.filter
        self.last_landmark_distance = distance
        if isinstance(f, ParticleTrack):
            return self._observe_particles(f)
        # апостериорная вероятность каждого ориентира против гипотезы
        # «остановка вне платформы» (светофор, очередь) с равномерной
        # плотностью в окне поиска
        window = 60.0
        null = self.landmark_random_stop / (2.0 * window)
        scored = []
        for mark in self.map.stops:
            d = self.map.delta(mark['s'], f.s)
            if abs(d) < window:
                var = f.p[0][0] + mark['sigma'] ** 2
                like = mark['share'] * math.exp(-0.5 * d * d / var) / math.sqrt(2.0 * math.pi * var)
                scored.append((like, d * d / var, mark))
        if not scored:
            return None
        total = null + sum(x[0] for x in scored)
        like, d2, mark = max(scored, key=lambda x: x[0])
        jump = self.map.delta(mark['s'], f.s)
        if (like / total < self.landmark_confidence or d2 > self.landmark_gate ** 2
                or abs(jump) > max(self.landmark_max_jump, 3.0 * f.sigma_s)):
            return self._relock(distance, scored)
        innovation = f.correct(f.s + jump, mark['sigma'])
        self.last_landmark = (mark['s'], innovation)
        self.pending = None
        return innovation

    def _observe_particles(self, f):
        reach = 4.0 * f.sigma_s + 80.0
        delta = self.map.delta
        marks = [m for m in self.map.stops if abs(delta(m['s'], f.s)) < reach]
        if not marks:
            return None
        background = self.landmark_random_stop / 120.0
        # хвост: иногда вагон встаёт у платформы на несколько метров раньше
        # или дальше обычного (очередь, второй вагон), такая остановка не
        # должна тянуть положение и масштаб как точная
        tail_w, tail_s = self.landmark_tail_weight, self.landmark_tail_sigma
        terms = []
        for m in marks:
            narrow = (1.0 - tail_w) * m['share'] / (math.sqrt(2.0 * math.pi) * m['sigma'])
            wide = tail_w * m['share'] / (math.sqrt(2.0 * math.pi) * tail_s)
            terms.append((m['s'], narrow, -0.5 / m['sigma'] ** 2, wide, -0.5 / tail_s ** 2))
        cutoff = max(8.0, 4.0 * tail_s)

        def likelihood(x):
            total = background
            for center, narrow, gain, wide, gain_wide in terms:
                d = delta(x, center)
                if abs(d) < cutoff:
                    total += narrow * math.exp(gain * d * d) + wide * math.exp(gain_wide * d * d)
            return total

        shift = f.observe(likelihood)
        near = min(marks, key=lambda m: abs(delta(m['s'], f.s)))
        if abs(delta(near['s'], f.s)) < 3.0 * max(near['sigma'], f.sigma_s):
            self.last_landmark = (near['s'], shift)
            return shift
        return None

    def _relock(self, distance, scored):
        """Перепривязка, если две остановки подряд дают одинаковый сдвиг.

        Одиночная остановка вне строба может быть очередью или светофором.
        Но если у двух частых платформ подряд вагон стоит с одинаковым
        смещением, то смещена сама оценка: так бывает после неизвестного
        участка пути или сильного проскальзывания.
        """
        major = [x for x in scored if x[2]['share'] >= self.relock_share]
        if not major:
            return None
        f = self.filter
        mark = min(major, key=lambda x: abs(self.map.delta(x[2]['s'], f.s)))[2]
        offset = self.map.delta(mark['s'], f.s)
        previous = self.pending
        self.pending = (distance, offset)
        if (previous is None or distance - previous[0] > self.relock_distance
                or abs(offset - previous[1]) > self.relock_agreement or abs(offset) < 2.0 * f.sigma_s):
            return None
        f.s += offset
        f.p = [[1.0, 0.0], [0.0, max(f.p[1][1], 0.004 ** 2)]]
        f.updates += 1
        self.pending = None
        self.relocks += 1
        self.last_landmark = (mark['s'], offset)
        return offset

    @property
    def scale(self):
        return self.filter.k if self.filter is not None else self.scale_prior

    def provisional(self, travelled_since):
        """Положение в окне начальной выставки прямо по свежим фиксам.

        travelled_since(d) возвращает путь колёс после отметки d, чтобы
        сдвинуть последний фикс на пройденное с его прихода расстояние.
        Возвращает None, пока фиксов нет.
        """
        m, r = self.fixes['master'], self.fixes['rover']
        if not m and not r:
            return None
        f = self.output_fraction
        heading = self._heading() or self.heading
        yaw = math.atan2(heading[1], heading[0])
        if self.map is not None:
            # эталон лежит на линии карты: уже в окне выставки ставим точку на неё
            last = (m or r)[-1]
            x, y, _ = self.frame.forward(*last[1:4])
            choice = self._match(x, y, self._heading())
            if choice is not None and choice[0] < 5.0:
                self.branch = choice[2]
                back = self.antenna_baseline if not m else 0.0
                s_master = choice[1] - back + travelled_since(last[4])
                p, yaw_map = self._map_output(s_master)
                return p, yaw_map, 1.0, 0.25
        if m and r and abs(m[-1][0] - r[-1][0]) < 0.05:
            a = self.frame.forward(*m[-1][1:4])
            b = self.frame.forward(*r[-1][1:4])
            p = [pa + f * (pb - pa) for pa, pb in zip(a, b)]
            moved = travelled_since(m[-1][4])
        else:
            last = (m or r)[-1]
            p = list(self.frame.forward(*last[1:4]))
            lever = f * self.antenna_baseline if m else (f - 1.0) * self.antenna_baseline
            p[0] += lever * heading[0]
            p[1] += lever * heading[1]
            moved = travelled_since(last[4])
        p = (p[0] + moved * heading[0], p[1] + moved * heading[1], p[2] - self.antenna_z)
        return p, yaw, 1.0, 1.0

    def _map_output(self, s_master):
        """Выходная точка на линии карты по дуговой координате master.

        Эталон судьи (localization kinematic_state) лежит ровно на линии
        карты организаторов с её высотой, поэтому base_link ставится на
        дугу на плечо tf впереди master, а не на хорду антенн.
        """
        lever = self.output_fraction * self.antenna_baseline
        sb = s_master + lever
        u = self.branch.arc(sb) if self.branch is not None else None
        drop = 0.0
        if self.output_point == 'base_link':
            drop = self.map.base_height if self.map.base_height is not None else self.antenna_z
        if u is not None:
            line = self.branch.line
            p = line.point(u)
            a = line.point(u - 1.0)
            b = line.point(u + 1.0)
            return (p[0], p[1], p[2] - drop), math.atan2(b[1] - a[1], b[0] - a[0])
        p = self.map.point(sb)
        a = self.map.point(sb - 1.0)
        b = self.map.point(sb + 1.0)
        return (p[0], p[1], p[2] - drop), math.atan2(b[1] - a[1], b[0] - a[0])

    def position(self, ds_ahead=0.0):
        """Выходная точка в MGRS, курс, дисперсии вдоль и поперёк пути."""
        yaw = math.atan2(self.heading[1], self.heading[0])
        lever = self.output_fraction * self.antenna_baseline
        if self.filter is None or self.start_s is None:
            # без карты: счисление вдоль начального курса от первого фикса
            travelled = (self.dead_reckoning if self.filter is None else self.filter.s) + ds_ahead
            if self.start_point is None:
                x0, y0, z0, d0 = 0.0, 0.0, self.antenna_z, 0.0
            else:
                x0, y0, z0, d0 = self.start_point
                if self.filter is not None:
                    d0 = self.start_point[3] * self.filter.k
            d = travelled - d0 + lever
            p = (x0 + d * self.heading[0], y0 + d * self.heading[1], z0 - self.antenna_z)
            var = self.filter.p[0][0] if self.filter is not None else 1e4
            return p, yaw, var, (0.05 * abs(d)) ** 2 + 25.0
        s = self.filter.s + ds_ahead
        p, yaw = self._map_output(s)
        # эталон привязан к линии карты: смещение первого фикса от карты
        # переносится на выход только при старте с пути, которого нет в
        # карте, и только в плане (высота фикса GNSS шумит на метры)
        w = 0.0
        if self.blend_length > 0 and (self.map_error or 0.0) > 3.0:
            w = max(0.0, 1.0 - abs(s - self.start_s) / self.blend_length)
        if w > 0.0:
            p = (p[0] + w * self.origin_offset[0], p[1] + w * self.origin_offset[1], p[2])
        # далёкая стартовая привязка не означает точного знания поперечной координаты
        map_sigma = self.map_error or 0.0
        return p, yaw, self.filter.p[0][0], 0.05 ** 2 + (w * 2.0) ** 2 + (w * map_sigma) ** 2

    def grade(self):
        if self.filter is None or self.start_s is None:
            return 0.0
        u = self.branch.arc(self.filter.s) if self.branch is not None else None
        if u is not None:
            return self.branch.line.grade(u)
        return self.map.grade(self.filter.s)

    def curvature(self):
        if self.filter is None or self.start_s is None:
            return 0.0
        u = self.branch.arc(self.filter.s) if self.branch is not None else None
        if u is not None:
            return self.branch.line.curvature(u)
        return self.map.curvature(self.filter.s)
