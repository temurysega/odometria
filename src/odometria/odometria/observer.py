"""Наблюдатель продольной скорости с учётом проскальзывания.

Состояние фильтра Калмана x = [v, b]: скорость по колёсам и поправка модели.
Прогноз:   v' = a_model(u_e, v, i) + b,  b' = w
Измерение: скорость передней или задней тележки, z = v + e.

Каждое измерение проходит проверки:
1. конечность и допустимый диапазон;
2. темп изменения колеса: боксование разгоняет колесо быстрее, чем может
   разогнаться вагон при текущей тяге, юз тормозит быстрее физического
   предела даже для экстренного торможения рельсовыми тормозами;
3. согласие с другой тележкой;
4. невязка с прогнозом в пределах адаптивного строба.
Две тележки, плотно согласные между собой и с физически возможным темпом,
принимаются даже вопреки модели: экстренное торможение не отражается в
позиции контроллера, а два независимых датчика редко ошибаются одинаково.
При расхождении тележек выбирается та, которую режим не может исказить:
при торможении и выбеге большая (юз занижает), при тяге меньшая (боксование
завышает); её вес снижен, так как проскальзывать могут обе. Отбракованная
тележка возвращается после попадания в узкую полосу около прогноза или
после согласия с принятой тележкой (гистерезис). Если обе отбракованы
дольше resync_after, фильтр пересинхронизируется по ним.
"""
from dataclasses import dataclass
import math


@dataclass
class ObserverConfig:
    wheel_scale: float = 1.0 / 3.6
    sigma_wheel: float = 0.03
    sigma_model: float = 0.35
    sigma_bias: float = 0.02
    bias_limit: float = 0.8
    pair_abs: float = 0.25
    pair_rel: float = 0.03
    sigma_disagree: float = 0.2
    pair_tight: float = 0.12
    pair_tight_rel: float = 0.01
    gate_sigma: float = 4.0
    gate_min: float = 0.35
    recover_band: float = 0.2
    rise_margin: float = 1.6
    max_decel: float = 6.0
    stale_after: float = 0.35
    resync_after: float = 4.0
    max_speed: float = 30.0
    min_speed: float = -3.0
    max_step: float = 0.1

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f'{name} должен быть конечным')
        if self.wheel_scale <= 0 or self.sigma_wheel <= 0 or self.max_step <= 0:
            raise ValueError('масштаб, шум и шаг должны быть положительными')


class Bogie:
    __slots__ = ('value', 'stamp', 'slipping', 'since', 'ratio', 'state', 'rejected')

    def __init__(self):
        self.value = None
        self.stamp = None
        self.slipping = False
        self.since = None
        self.ratio = 0.0
        self.state = 'нет данных'
        self.rejected = 0


class VelocityObserver:
    def __init__(self, model, config=None):
        self.model = model
        self.cfg = config or ObserverConfig()
        self.t = None
        self.v = 0.0
        self.b = 0.0
        self.p = [[1.0, 0.0], [0.0, 0.05 ** 2]]
        self.ue = 0.0
        self.command = 0
        self.grade = 0.0
        self.distance = 0.0
        self.accel = 0.0
        self.last_accept = None
        self.model_only_time = 0.0
        self.bogies = {'front': Bogie(), 'rear': Bogie()}

    def set_command(self, position):
        self.command = int(max(-15, min(15, position)))

    def model_acceleration(self, v=None):
        v = self.v if v is None else v
        a = self.model.acceleration(self.ue, v, self.grade, self.command)
        if abs(v) < 0.05 and a == 0.0:
            return 0.0
        return a + self.b

    def predict(self, t):
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0.0:
            return
        cfg = self.cfg
        q_v = cfg.sigma_model ** 2
        q_b = cfg.sigma_bias ** 2
        while dt > 1e-9:
            h = min(dt, cfg.max_step)
            self.ue = self.model.actuator(self.ue, self.command, h)
            a = self.model_acceleration()
            v0 = self.v
            v1 = v0 + a * h
            if v0 >= 0.0 > v1:
                v1 = 0.0
            self.v = max(cfg.min_speed, min(cfg.max_speed, v1))
            self.distance += 0.5 * (v0 + self.v) * h
            self.accel = a
            p = self.p
            pvv = p[0][0] + 2 * h * p[0][1] + h * h * p[1][1] + q_v * h
            pvb = p[0][1] + h * p[1][1]
            pbb = p[1][1] + q_b * h
            self.p = [[pvv, pvb], [pvb, pbb]]
            dt -= h
        if self.last_accept is not None and t - self.last_accept > 0.3:
            self.model_only_time = t - self.last_accept
        self.t = t

    def _kalman(self, z, r):
        p = self.p
        s = p[0][0] + r
        k0 = p[0][0] / s
        k1 = p[0][1] / s
        innovation = z - self.v
        self.v += k0 * innovation
        self.b = max(-self.cfg.bias_limit, min(self.cfg.bias_limit, self.b + k1 * innovation))
        self.p = [[(1 - k0) * p[0][0], (1 - k0) * p[0][1]],
                  [(1 - k0) * p[0][1], p[1][1] - k1 * p[0][1]]]

    def update(self, side, t, raw):
        """Обрабатывает измерение тележки, возвращает его статус."""
        cfg = self.cfg
        bogie = self.bogies[side]
        other = self.bogies['rear' if side == 'front' else 'front']
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            raw = float('nan')
        if not math.isfinite(raw):
            bogie.state = 'некорректное значение'
            return bogie.state
        z = raw * cfg.wheel_scale
        if not cfg.min_speed <= z <= cfg.max_speed:
            bogie.state = 'вне диапазона'
            return bogie.state
        if self.t is None:
            self.predict(t)
            self.v = z
            self.p = [[cfg.sigma_wheel ** 2, 0.0], [0.0, self.p[1][1]]]
            self._accept(bogie, z, t, 'норма')
            return bogie.state
        self.predict(t)
        a_pred = self.model_acceleration()
        implausible = False
        if bogie.stamp is not None and 0.03 < t - bogie.stamp < 0.6:
            rate = (z - bogie.value) / (t - bogie.stamp)
            implausible = rate > max(a_pred, 0.0) + cfg.rise_margin or rate < -cfg.max_decel
        pair_ok = None
        pair_tight = False
        other_pred = None
        if other.stamp is not None and 0.0 <= t - other.stamp < cfg.stale_after:
            other_pred = other.value + a_pred * (t - other.stamp)
            gap = abs(z - other_pred)
            pair_ok = gap < cfg.pair_abs + cfg.pair_rel * abs(self.v)
            pair_tight = gap < cfg.pair_tight + cfg.pair_tight_rel * abs(self.v)
        r = cfg.sigma_wheel ** 2
        spread = math.sqrt(self.p[0][0] + r)
        innovation = z - self.v
        if bogie.slipping:
            consistent = abs(innovation) < max(cfg.recover_band, 2.0 * spread)
        else:
            consistent = abs(innovation) < max(cfg.gate_min, cfg.gate_sigma * spread)
        if pair_tight:
            consistent = True
        consistent = consistent and not implausible
        disagree = pair_ok is not None and not pair_ok
        traction = self.ue > 0.05
        if z == 0.0 and traction and self.v > 0.5:
            # ровный ноль под тягой на ходу: отказ датчика, а не юз
            consistent = False
        elif disagree:
            # юз занижает колесо, боксование завышает: верим той тележке,
            # которую текущий режим не может исказить в её сторону, и только
            # если она сама проходит строб невязки
            better = z < other_pred if traction else z > other_pred
            consistent = consistent and better
        bogie.ratio = innovation / max(abs(self.v), 1.0)
        if consistent:
            self._accept(bogie, z, t, 'норма', cfg.sigma_disagree ** 2 if disagree else None)
            return bogie.state
        blind = t - (self.last_accept if self.last_accept is not None else t)
        if blind > cfg.resync_after and not implausible and not disagree:
            # оценка слишком долго держалась только на модели
            self.v = z if other_pred is None else 0.5 * (z + other_pred)
            self.p = [[r, 0.0], [0.0, self.p[1][1]]]
            other.slipping = False
            self._accept(bogie, z, t, 'пересинхронизация')
            return bogie.state
        if not bogie.slipping:
            bogie.slipping = True
            bogie.since = t
        bogie.rejected += 1
        bogie.value, bogie.stamp = z, t
        bogie.state = ('юз' if innovation < 0 and self.command <= 0 else
                       'боксование' if innovation > 0 and self.command > 0 else
                       'расхождение с моделью')
        return bogie.state

    def _accept(self, bogie, z, t, state, variance=None):
        self._kalman(z, self.cfg.sigma_wheel ** 2 if variance is None else variance)
        bogie.slipping = False
        bogie.since = None
        bogie.value, bogie.stamp = z, t
        bogie.state = state
        self.last_accept = t
        self.model_only_time = 0.0

    @property
    def sigma_v(self):
        return math.sqrt(max(self.p[0][0], 0.0))

    @property
    def slip_detected(self):
        return any(b.slipping for b in self.bogies.values())

    def health(self, t):
        fresh = [b for b in self.bogies.values()
                 if b.stamp is not None and t - b.stamp < self.cfg.stale_after]
        if not fresh:
            return 'нет колёс'
        if self.model_only_time > 0.3:
            return 'только модель'
        if self.slip_detected:
            return 'проскальзывание'
        return 'норма'
