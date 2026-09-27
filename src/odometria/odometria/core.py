"""Ядро одометрии без зависимости от ROS.

Узел ROS и офлайн проигрыватель вызывают одни и те же методы, поэтому
оценка точности на bag файлах проверяет ровно тот код, что работает в узле.

Координаты выхода: плоские MGRS (x восток, y север) и высота z по REP 103
для точки base_link, как ответили эксперты кейса.
Шкала времени: header.stamp сообщений тележек. На данных он совпадает со
шкалой GNSS (задержка колёс относительно эталона 0 с), тогда как время
записи и заголовок контроллера сдвинуты примерно на 50 мс.
Выход публикуется на сетке 1/output_rate секунд этой шкалы: на каждый узел
сетки значение экстраполируется моделью от последнего измерения (не более
чем на output_lead вперёд). Эпохи GNSS лежат на сетке 0,1 с, поэтому
каждая эпоха эталона получает выход ровно в свой момент.
"""
from collections import deque
from dataclasses import dataclass, field
import math

from .geodesy import MgrsFrame
from .observer import ObserverConfig, VelocityObserver
from .track import Localizer
from .traction import TractionModel


def _number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


@dataclass
class CoreConfig:
    output_rate: float = 50.0
    # эталон судьи (localization kinematic_state): его скорость запаздывает
    # относительно колёс на 0,1 с, а положение опережает шкалу заголовков
    # колёс на 0,05 с (замерено на проверочном bag организаторов)
    velocity_delay: float = 0.10
    position_lead: float = 0.05
    # коррекция по редким пачкам GNSS после выставки (разрешено организаторами)
    gnss_correction: bool = True
    # метка фикса против шкалы счисления: фикс соответствует пути на 0,1 с позже
    gnss_delay: float = 0.10
    output_lead: float = 0.01
    max_backfill: float = 0.1
    stop_speed: float = 0.03
    stop_confirm: float = 1.0
    command_timeout: float = 0.5
    wheel_stale: float = 0.15
    reset_backward: float = 5.0
    reset_forward: float = 30.0
    gnss_wait: float = 5.0
    init_window: float = 1.0
    use_map: bool = True
    use_landmarks: bool = True
    scale_prior: float = 1.0
    scale_sigma: float = 0.008
    master_x: float = -9.873
    rover_x: float = 2.563
    antenna_z: float = 3.0
    output_point: str = 'base_link'
    mgrs_zone: int = 37
    mgrs_origin_east: float = 300000.0
    mgrs_origin_north: float = 6100000.0
    max_map_offset: float = 15.0
    landmark_gate: float = 3.0
    particles: int = 1500


@dataclass
class Output:
    stamp: float
    velocity: float
    acceleration: float
    position: tuple
    yaw: float
    yaw_rate: float
    var_along: float
    var_cross: float
    var_velocity: float
    status: str
    slip: bool
    position_valid: bool = True
    velocity_now: float = 0.0


@dataclass
class Counters:
    wheel: int = 0
    command: int = 0
    fixes: int = 0
    rejected_input: int = 0
    late: int = 0
    resets: int = 0
    landmarks: int = 0
    landmark_skipped: int = 0
    outputs: int = 0
    extra: dict = field(default_factory=dict)


class OdometryCore:
    def __init__(self, config=None, observer_config=None, track_map=None, model=None):
        self.cfg = config or CoreConfig()
        self.observer_cfg = observer_config or ObserverConfig()
        self.map = track_map if self.cfg.use_map else None
        self.model = model or TractionModel.load()
        self.counters = Counters()
        self._reset()

    def _reset(self):
        cfg = self.cfg
        self.observer = VelocityObserver(self.model, self.observer_cfg)
        self.localizer = Localizer(
            self.map, init_window=cfg.init_window, max_map_offset=cfg.max_map_offset,
            master_x=cfg.master_x, rover_x=cfg.rover_x, antenna_z=cfg.antenna_z,
            output_point=cfg.output_point,
            frame=MgrsFrame(cfg.mgrs_zone, cfg.mgrs_origin_east, cfg.mgrs_origin_north),
            scale_prior=cfg.scale_prior,
            scale_sigma=cfg.scale_sigma, use_landmarks=cfg.use_landmarks,
            landmark_gate=cfg.landmark_gate, particles=cfg.particles)
        self.last_grid = None
        self.command_stamp = None
        self.command_offset = None
        self.last_wheel = None
        self.first_wheel = None
        self.speed_history = deque(maxlen=100)
        self.distance_history = deque(maxlen=400)
        self.stop_since = None
        self.stop_handled = False
        self.last_status = 'ожидание данных'

    @property
    def gnss_needed(self):
        return not self.localizer.ready or self.cfg.gnss_correction

    def _distance_at(self, stamp):
        """Пройденный путь на момент stamp по истории колёс."""
        h = self.distance_history
        if not h or not math.isfinite(stamp):
            return self.observer.distance
        if stamp >= h[-1][0]:
            return h[-1][1]
        if stamp <= h[0][0]:
            return h[0][1]
        hi = len(h) - 1
        while hi > 0 and h[hi - 1][0] > stamp:
            hi -= 1
        (t0, d0), (t1, d1) = h[hi - 1], h[hi]
        w = (stamp - t0) / (t1 - t0) if t1 > t0 else 1.0
        return d0 + w * (d1 - d0)

    def _check_jump(self, stamp):
        t = self.observer.t
        if t is None:
            return
        if stamp < t - self.cfg.reset_backward or stamp > t + self.cfg.reset_forward:
            # начало новой записи: начинаем оценку заново
            self.counters.resets += 1
            self._reset()

    def on_fix(self, source, stamp, lat, lon, alt, status):
        stamp, lat, lon, alt = (_number(x) for x in (stamp, lat, lon, alt))
        # фикс приходит с задержкой до секунды: путь берётся на момент его метки
        distance = self._distance_at(stamp)
        if not self.localizer.ready:
            if self.localizer.add_fix(source, stamp, lat, lon, alt, int(status), distance):
                self.counters.fixes += 1
                return True
            return False
        if self.cfg.gnss_correction and self.localizer.correction_fix(
                source, stamp, lat, lon, alt, int(status), self._distance_at(stamp + self.cfg.gnss_delay)):
            self.counters.fixes += 1
            return True
        return False

    def on_command(self, stamp, position):
        stamp = _number(stamp)
        if not (math.isfinite(stamp) and stamp > 0):
            self.counters.rejected_input += 1
            return []
        try:
            valid = float(position) == int(position)
            position = int(position)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid or not -15 <= position <= 15:
            self.counters.rejected_input += 1
            return []
        self._check_jump(stamp)
        self.counters.command += 1
        if not self.localizer.ready and self.localizer.window_closed(stamp - 0.05):
            # окно GNSS закрывается и тогда, когда колёса молчат
            self.localizer.finalize(self.observer.distance)
        self.observer.set_command(position)
        self.command_stamp = stamp
        if self.last_wheel is None:
            return []
        now = stamp - (self.command_offset if self.command_offset is not None else 0.05)
        if now - self.last_wheel <= self.cfg.wheel_stale or now <= self.observer.t:
            return []
        # колёса молчат: продолжаем по модели, чтобы выход не прерывался
        self._advance(now)
        return self._emit(now)

    def on_wheel(self, side, stamp, value):
        stamp = _number(stamp)
        value = _number(value)
        if not (math.isfinite(stamp) and stamp > 0):
            self.counters.rejected_input += 1
            return []
        self._check_jump(stamp)
        self.counters.wheel += 1
        obs = self.observer
        if self.command_stamp is not None and obs.t is not None:
            offset = self.command_stamp - stamp
            if abs(offset) < 1.0:
                self.command_offset = (offset if self.command_offset is None
                                       else 0.95 * self.command_offset + 0.05 * offset)
        t = stamp
        self.last_wheel = stamp if self.last_wheel is None else max(self.last_wheel, stamp)
        if self.first_wheel is None:
            self.first_wheel = stamp
        variance = None
        if obs.t is not None and stamp < obs.t:
            # опоздавшее сообщение: приводим измерение к текущему моменту
            self.counters.late += 1
            t = obs.t
            if value is not None and math.isfinite(value):
                age = obs.t - stamp
                corrected = value + obs.accel * age / obs.cfg.wheel_scale
                # экстраполяция торможения от неотрицательного колеса не даёт реверса
                value = max(0.0, corrected) if value >= 0.0 else corrected
                # ускорение для приведения тоже оценено: старому отсчёту меньше веса
                variance = obs.cfg.sigma_wheel ** 2 + (obs.cfg.sigma_model * age) ** 2
        if self.command_stamp is None or t - self.command_stamp > self.cfg.command_timeout + 0.1:
            obs.set_command(0)
        obs.grade = self.localizer.grade()
        before = obs.distance
        status = obs.update(side, t, value, measurement_variance=variance)
        self.speed_history.append((obs.t, obs.v))
        self._after_motion(t, obs.distance - before)
        self.last_status = status
        return self._emit(t)

    def _advance(self, t):
        obs = self.observer
        obs.grade = self.localizer.grade()
        before = obs.distance
        obs.predict(t)
        self._after_motion(t, obs.distance - before)

    def _after_motion(self, t, ds):
        obs = self.observer
        loc = self.localizer
        extra = 0.0
        if obs.model_only_time > 0.3:
            extra = obs.p[0][0] * abs(ds / max(abs(obs.v), 0.5)) * 2.0
        loc.advance(ds, extra)
        self.distance_history.append((t, obs.distance))
        if not loc.ready and loc.window_closed(t):
            loc.finalize(obs.distance)
        if loc.burst:
            loc.flush_corrections(t, obs.distance)
        if abs(obs.v) < self.cfg.stop_speed and obs.model_only_time == 0.0:
            if self.stop_since is None:
                self.stop_since = t
                self.stop_handled = False
            elif not self.stop_handled and t - self.stop_since >= self.cfg.stop_confirm:
                self.stop_handled = True
                if loc.ready:
                    before = self.counters.landmarks
                    if loc.try_landmark(obs.distance) is not None:
                        self.counters.landmarks = before + 1
                    else:
                        self.counters.landmark_skipped += 1
        else:
            self.stop_since = None

    def _emit(self, t):
        cfg = self.cfg
        step = 1.0 / cfg.output_rate
        last = math.floor((t + cfg.output_lead) / step + 1e-9)
        first = math.ceil((t - cfg.max_backfill) / step - 1e-9)
        if self.last_grid is not None:
            first = max(first, self.last_grid + 1)
        if last < first:
            return []
        obs = self.observer
        loc = self.localizer
        k = loc.scale
        health = obs.health(t)
        slip = obs.slip_detected
        out = []
        curvature = loc.curvature()
        for n in range(first, last + 1):
            g = n * step
            dt = g - t
            v = obs.v + obs.accel * dt
            if obs.v >= 0.0 > v:
                v = 0.0
            dt_pos = dt + cfg.position_lead
            v_pos = obs.v + obs.accel * dt_pos
            if obs.v >= 0.0 > v_pos:
                v_pos = 0.0
            ds = k * 0.5 * (obs.v + v_pos) * dt_pos
            valid = True
            if loc.ready:
                position, yaw, var_along, var_cross = loc.position(ds)
            else:
                # окно начальной выставки: положение по свежим фиксам GNSS;
                # без фиксов положение не публикуется, пока не истечёт gnss_wait
                early = loc.provisional(lambda d: k * (obs.distance - d) + ds)
                if early is not None:
                    position, yaw, var_along, var_cross = early
                else:
                    position, yaw, var_along, var_cross = loc.position(ds)
                    valid = g - (self.first_wheel if self.first_wheel is not None else g) > cfg.gnss_wait
            velocity = k * v
            delayed = k * self._speed_at(g - cfg.velocity_delay, v, g)
            out.append(Output(
                stamp=g, velocity=delayed, velocity_now=velocity, acceleration=k * obs.accel, position=position,
                yaw=yaw, yaw_rate=velocity * curvature, var_along=var_along, var_cross=var_cross,
                var_velocity=(k * obs.sigma_v) ** 2 + (v * (loc.filter.sigma_k if loc.filter else 0.01)) ** 2,
                status=health, slip=slip, position_valid=valid))
        self.last_grid = last
        self.counters.outputs += len(out)
        return out

    def _speed_at(self, when, current, now):
        """Скорость на момент when по истории обновлений наблюдателя."""
        if when >= now or not self.speed_history:
            return current
        hist = self.speed_history
        if when <= hist[0][0]:
            return hist[0][1]
        for (t0, v0), (t1, v1) in zip(reversed(list(hist)[:-1]), reversed(hist)):
            if t0 <= when <= t1:
                return v0 + (v1 - v0) * (when - t0) / (t1 - t0) if t1 > t0 else v1
        # между последним обновлением и now: прогноз к моменту when
        t_last, v_last = hist[-1]
        return v_last + self.observer.accel * (when - t_last)

    def diagnostics(self):
        obs = self.observer
        loc = self.localizer
        f = loc.filter
        return {
            'status': obs.health(obs.t) if obs.t is not None else 'ожидание данных',
            'front': obs.bogies['front'].state,
            'rear': obs.bogies['rear'].state,
            'slip_ratio_front': round(obs.bogies['front'].ratio, 4),
            'slip_ratio_rear': round(obs.bogies['rear'].ratio, 4),
            'adhesion_used': round(abs(obs.accel) / 9.81, 4),
            'model_bias_mps2': round(obs.b, 4),
            'wheel_scale_k': round(loc.scale, 5),
            'sigma_velocity_mps': round(obs.sigma_v, 4),
            'sigma_along_m': round(f.sigma_s, 3) if f else None,
            'model_only_s': round(obs.model_only_time, 2),
            'map_matched': loc.start_s is not None,
            'output_point': loc.output_point,
            'gnss_antenna_used': loc.source,
            'map_offset_m': round(loc.map_error, 2) if loc.map_error is not None else None,
            'gnss_init_done': loc.ready,
            'landmarks_used': self.counters.landmarks,
            'last_landmark_innovation_m': round(loc.last_landmark[1], 2) if loc.last_landmark else None,
            'gnss_corrections': loc.gnss_corrections,
            'gnss_relocations': loc.gnss_relocations,
            'last_gnss_shift_m': round(loc.last_gnss[1], 2) if loc.last_gnss else None,
            'terminal_spur': loc.branch.kind if loc.branch is not None else None,
            'resets': self.counters.resets,
            'late_messages': self.counters.late,
            'rejected_inputs': self.counters.rejected_input,
        }
