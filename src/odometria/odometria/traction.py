"""Нелинейная модель тягового привода и продольной динамики.

    du_e/dt = (u / 15 - u_e) / tau
    a = T(u_e, v) - g * i(s) + b

T(u_e, v) таблица удельной силы тяги или торможения (м/с²), откалиброванная
по разрешённым данным (tools/fit_traction.py). Зависимость от скорости
отражает ограничение мощности привода на высокой скорости и угасание
электрического торможения на малой. Сопротивление движению (качение,
аэродинамика) входит в таблицу, так как она подобрана по фактическому
ускорению. Уклон i(s) берётся из высот карты, b оценивает фильтр онлайн
(масса, ветер, состояние рельсов).
"""
from bisect import bisect_right
import json
import math
from pathlib import Path

DATA = Path(__file__).with_name('data')


class TractionModel:
    def __init__(self, u_knots, v_knots, table, tau, gravity=9.80665):
        if len(u_knots) < 2 or len(v_knots) < 2:
            raise ValueError('таблица тяги должна иметь не менее двух узлов по каждой оси')
        if len(table) != len(u_knots) or any(len(r) != len(v_knots) for r in table):
            raise ValueError('размер таблицы не совпадает с узлами')
        self.u_knots = [float(x) for x in u_knots]
        self.v_knots = [float(x) for x in v_knots]
        self.table = [[float(x) for x in row] for row in table]
        self.tau = float(tau)
        self.gravity = float(gravity)
        if (not all(math.isfinite(x) for x in self.u_knots + self.v_knots) or
                any(b <= a for knots in (self.u_knots, self.v_knots)
                    for a, b in zip(knots, knots[1:])) or
                not all(math.isfinite(x) for row in self.table for x in row) or
                not math.isfinite(self.tau) or self.tau <= 0.0 or
                not math.isfinite(self.gravity) or self.gravity <= 0.0):
            raise ValueError('некорректные параметры таблицы тяги')

    @classmethod
    def load(cls, path=None):
        doc = json.loads(Path(path or DATA / 'traction_model.json').read_text(encoding='utf-8'))
        return cls(doc['u_knots'], doc['v_knots'], doc['table'], doc['tau_s'], doc.get('gravity', 9.80665))

    def actuator(self, state, command, dt):
        """Отклик привода на позицию контроллера за шаг dt."""
        if dt <= 0.0:
            return state
        target = max(-1.0, min(1.0, command / 15.0))
        return state + (target - state) * (1.0 - math.exp(-dt / self.tau))

    @staticmethod
    def _cell(x, knots):
        i = min(max(bisect_right(knots, x) - 1, 0), len(knots) - 2)
        w = (x - knots[i]) / (knots[i + 1] - knots[i])
        return i, min(1.0, max(0.0, w))

    def drive(self, ue, v):
        """Табличная тяга или торможение без уклона."""
        iu, wu = self._cell(ue, self.u_knots)
        iv, wv = self._cell(min(max(v, 0.0), self.v_knots[-1]), self.v_knots)
        t = self.table
        # Позиция контроллера не является командой тормозного усилия:
        # на записях с u=-15 и согласными тележками/GNSS вагон иногда
        # действительно ускоряется. Ограничивать знак здесь нельзя.
        return ((1 - wu) * (1 - wv) * t[iu][iv] + wu * (1 - wv) * t[iu + 1][iv]
                + (1 - wu) * wv * t[iu][iv + 1] + wu * wv * t[iu + 1][iv + 1])

    def acceleration(self, ue, v, grade=0.0, command=0):
        a = self.drive(ue, abs(v)) - self.gravity * grade
        if abs(v) < 0.05 and (command <= 0 or a < 0.0):
            # на месте тормоза удерживают вагон, назад он не скатывается
            return 0.0
        return a
