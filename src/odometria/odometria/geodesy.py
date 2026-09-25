"""Точные преобразования WGS84 без внешних зависимостей.

Локальная система координат: касательная плоскость ENU (восток, север, верх)
в точке начала, как GeographicLib LocalCartesian. Сферическое приближение
с радиусом 6371 км на широте Москвы занижает расстояния по долготе на 0,34 %,
то есть до 15 м на длине линии, поэтому здесь используется эллипсоид.
"""
import math

A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)


def to_ecef(lat, lon, alt):
    la = math.radians(lat)
    lo = math.radians(lon)
    sin_la = math.sin(la)
    n = A / math.sqrt(1.0 - E2 * sin_la * sin_la)
    return ((n + alt) * math.cos(la) * math.cos(lo),
            (n + alt) * math.cos(la) * math.sin(lo),
            (n * (1.0 - E2) + alt) * sin_la)


class LocalFrame:
    """Касательная плоскость ENU в заданной точке WGS84."""

    def __init__(self, lat, lon, alt):
        self.origin = (lat, lon, alt)
        self.ecef0 = to_ecef(lat, lon, alt)
        la = math.radians(lat)
        lo = math.radians(lon)
        sl, cl = math.sin(la), math.cos(la)
        so, co = math.sin(lo), math.cos(lo)
        self.rot = ((-so, co, 0.0),
                    (-sl * co, -sl * so, cl),
                    (cl * co, cl * so, sl))

    def from_ecef(self, x, y, z):
        d = (x - self.ecef0[0], y - self.ecef0[1], z - self.ecef0[2])
        r = self.rot
        return (r[0][0] * d[0] + r[0][1] * d[1] + r[0][2] * d[2],
                r[1][0] * d[0] + r[1][1] * d[1] + r[1][2] * d[2],
                r[2][0] * d[0] + r[2][1] * d[1] + r[2][2] * d[2])

    def to_ecef(self, e, n, u):
        r = self.rot
        return (self.ecef0[0] + r[0][0] * e + r[1][0] * n + r[2][0] * u,
                self.ecef0[1] + r[0][1] * e + r[1][1] * n + r[2][1] * u,
                self.ecef0[2] + r[0][2] * e + r[1][2] * n + r[2][2] * u)

    def from_geodetic(self, lat, lon, alt):
        return self.from_ecef(*to_ecef(lat, lon, alt))

    def transfer(self, other, e, n, u):
        """Переводит точку из этой системы в систему other."""
        return other.from_ecef(*self.to_ecef(e, n, u))
