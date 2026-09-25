"""Карта пути, начальная привязка и вдольпутевой фильтр.

Карта: осевая master антенны (контур из двух направлений, шаг 1 м) и
ориентиры остановок, построенные офлайн (tools/build_map.py).

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

from .geodesy import LocalFrame

DATA = Path(__file__).with_name('data')


class TrackMap:
    def __init__(self, origin, points, stops=(), east_junction=None, closed=False):
        if len(points) < 2:
            raise ValueError('в карте меньше двух точек')
        self.frame = LocalFrame(*origin)
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
        self._smooth_z = self._robust_profile([p[2] for p in self.points])
        self._grid = {}
        for i, (x, y, _) in enumerate(self.points):
            self._grid.setdefault((int(x // 50), int(y // 50)), []).append(i)

    @classmethod
    def load(cls, path=None):
        doc = json.loads(Path(path or DATA / 'track_map.json').read_text(encoding='utf-8'))
        return cls(doc['origin_wgs84'], doc['points_enu'], doc.get('stops', ()),
                   doc.get('east_junction_s'), doc.get('closed', False))

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
        self.var_s = sum(self.weight[i] * (self.base[i] - s) ** 2 for i in best) / wsum
        self.var_k = sum(self.weight[i] * (self.scale[i] - k) ** 2 for i in best) / wsum
        self.cov_sk = sum(self.weight[i] * (self.base[i] - s) * (self.scale[i] - k) for i in best) / wsum
        self.s, self.k = s, k
        self.share = wsum


class Localizer:
    """Начальная привязка по первым секундам GNSS и выдача положения.

    GNSS принимается только в окне init_window после первого валидного фикса.
    Начало локальной ENU системы: первый валидный фикс master (при его
    отсутствии rover), что совпадает с началом эталонной траектории.
    """

    def __init__(self, track_map=None, init_window=1.0, max_map_offset=15.0,
                 antenna_baseline=12.4, blend_length=80.0, scale_prior=1.0, scale_sigma=0.008,
                 use_landmarks=True, landmark_gate=3.0, landmark_max_jump=12.0,
                 landmark_confidence=0.9, landmark_random_stop=0.15, particles=1500):
        self.map = track_map
        self.init_window = init_window
        self.max_map_offset = max_map_offset
        self.antenna_baseline = antenna_baseline
        self.blend_length = blend_length
        self.scale_prior = scale_prior
        self.scale_sigma = scale_sigma
        self.use_landmarks = use_landmarks
        self.landmark_gate = landmark_gate
        self.landmark_max_jump = landmark_max_jump
        self.landmark_confidence = landmark_confidence
        self.landmark_random_stop = landmark_random_stop
        self.particles = particles
        self.relock_share = 0.3
        self.relock_distance = 2000.0
        self.relock_agreement = 3.0
        self.pending = None
        self.relocks = 0
        self.fixes = {'master': [], 'rover': []}
        self.first_fix_time = None
        self.ready = False
        self.frame = None
        self.target = 'master'
        self.filter = None
        self.origin_offset = (0.0, 0.0, 0.0)
        self.start_s = None
        self.heading = (1.0, 0.0)
        self.map_error = None
        self.local_points = None
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
            self.frame = None
            return False
        self.target = source
        lat0, lon0, alt0 = fixes[0][1:4]
        self.frame = LocalFrame(lat0, lon0, alt0)
        heading = self._heading()
        if heading is not None:
            self.heading = heading
        if self.map is None:
            self.filter = AlongTrackFilter(distance, k=self.scale_prior, sigma_k=self.scale_sigma)
            self.start_s = None
            return True
        offset = self.antenna_baseline if source == 'rover' else 0.0
        estimates = []
        best = None
        for t, lat, lon, alt, d, _ in fixes:
            x, y, z = self.map.frame.from_geodetic(lat, lon, alt)
            choice = self._match(x, y, heading)
            if choice is None:
                continue
            dist, s_fix = choice
            estimates.append(s_fix - offset - d)
            if best is None:
                best = (dist, (x, y, z))
        if not estimates:
            self.filter = AlongTrackFilter(distance, k=self.scale_prior, sigma_k=self.scale_sigma)
            self.start_s = None
            self.map_error = None
            return True
        estimates.sort()
        base = estimates[len(estimates) // 2]
        self.map_error = best[0]
        self.start_s = base + distance
        # точность старта по статусу фикса: 2 RTK, 1 SBAS, 0 автономное решение
        status_sigma = {2: 0.3, 1: 1.5}.get(max(f[5] for f in fixes), 3.0)
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
        # точки карты переводятся в локальную систему эталона по мере надобности
        self.local_points = {}
        first = fixes[0]
        start_local = self.frame.from_geodetic(first[1], first[2], first[3])
        on_map = self._local_point(base + first[4] + offset)
        self.origin_offset = tuple(a - b for a, b in zip(start_local, on_map))
        return True

    def _heading(self):
        m, r = self.fixes['master'], self.fixes['rover']
        if m and r:
            frame = LocalFrame(*m[0][1:4])
            e, n, _ = frame.from_geodetic(*r[0][1:4])
            norm = math.hypot(e, n)
            if 5.0 < norm < 30.0:
                return e / norm, n / norm
        pts = m or r
        if len(pts) >= 2:
            frame = LocalFrame(*pts[0][1:4])
            e, n, _ = frame.from_geodetic(*pts[-1][1:4])
            norm = math.hypot(e, n)
            if norm > 3.0:
                return e / norm, n / norm
        return None

    def _match(self, x, y, heading):
        options = self.map.candidates(x, y, self.max_map_offset)
        if not options:
            # стоянка на пути, которого нет в карте (конечная с несколькими
            # тупиками): берём ближайший путь того же направления, смещение
            # до него плавно убирается на первых метрах движения
            options = self.map.candidates(x, y, 4.0 * self.max_map_offset)
            if heading is None or not options:
                return None
            options = [o for o in options if o[2] * heading[0] + o[3] * heading[1] > 0.7]
            if not options:
                return None
        if heading is not None:
            # на двухпутном участке соседний путь в 4 м, отсекаем встречное направление
            aligned = [o for o in options if o[2] * heading[0] + o[3] * heading[1] > 0.5]
            options = aligned or options
        dist, s, _, _ = min(options, key=lambda o: o[0])
        return dist, s

    def _local_vertex(self, i):
        cached = self.local_points.get(i)
        if cached is None:
            cached = self.map.frame.transfer(self.frame, *self.map.points[i])
            self.local_points[i] = cached
        return cached

    def _local_point(self, s):
        s = self.map.wrap(s)
        i = self.map._index(s)
        a, b = self._local_vertex(i), self._local_vertex(i + 1)
        span = self.map.s[i + 1] - self.map.s[i]
        w = (s - self.map.s[i]) / span if span > 0 else 0.0
        return tuple(pa + w * (pb - pa) for pa, pb in zip(a, b))

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
        terms = [(m['s'], m['share'] / (math.sqrt(2.0 * math.pi) * m['sigma']),
                  -0.5 / m['sigma'] ** 2) for m in marks]

        def likelihood(x):
            total = background
            for center, peak, gain in terms:
                d = delta(x, center)
                if abs(d) < 8.0:
                    total += peak * math.exp(gain * d * d)
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

    def position(self, ds_ahead=0.0):
        """Положение цели в локальной ENU, курс, дисперсии вдоль и поперёк."""
        if self.filter is None:
            d = self.dead_reckoning + ds_ahead
            return (d * self.heading[0], d * self.heading[1], 0.0), math.atan2(self.heading[1], self.heading[0]), 1e4, 1e4
        s = self.filter.s + ds_ahead
        if self.start_s is None:
            d = s
            yaw = math.atan2(self.heading[1], self.heading[0])
            return (d * self.heading[0], d * self.heading[1], 0.0), yaw, self.filter.p[0][0], (0.05 * abs(d)) ** 2 + 25.0
        offset = self.antenna_baseline if self.target == 'rover' else 0.0
        p = self._local_point(s + offset)
        w = max(0.0, 1.0 - abs(s - self.start_s) / self.blend_length) if self.blend_length > 0 else 0.0
        if w > 0.0:
            p = tuple(a + w * o for a, o in zip(p, self.origin_offset))
        a = self._local_point(s + offset - 1.0)
        b = self._local_point(s + offset + 1.0)
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        return p, yaw, self.filter.p[0][0], 0.05 ** 2 + (w * 2.0) ** 2

    def grade(self):
        if self.filter is None or self.start_s is None:
            return 0.0
        return self.map.grade(self.filter.s)

    def curvature(self):
        if self.filter is None or self.start_s is None:
            return 0.0
        return self.map.curvature(self.filter.s)
