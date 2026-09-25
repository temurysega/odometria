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


_N = F / (2.0 - F)
_A = A / (1.0 + _N) * (1.0 + _N ** 2 / 4.0 + _N ** 4 / 64.0)
_ALPHA = (_N / 2.0 - 2.0 * _N ** 2 / 3.0 + 5.0 * _N ** 3 / 16.0 + 41.0 * _N ** 4 / 180.0,
          13.0 * _N ** 2 / 48.0 - 3.0 * _N ** 3 / 5.0 + 557.0 * _N ** 4 / 1440.0,
          61.0 * _N ** 3 / 240.0 - 103.0 * _N ** 4 / 140.0,
          49561.0 * _N ** 4 / 161280.0)
_K0 = 0.9996


def utm_zone(lon):
    return int((lon + 180.0) // 6.0) + 1


def utm_forward(lat, lon, zone=None):
    """Широта и долгота WGS84 в UTM (ряды Крюгера, точность лучше 1 мм в зоне).

    Возвращает восток и север в метрах для северного полушария и номер зоны.
    """
    zone = zone or utm_zone(lon)
    lon0 = math.radians(6.0 * zone - 183.0)
    phi = math.radians(lat)
    dlon = math.radians(lon) - lon0
    c = 2.0 * math.sqrt(_N) / (1.0 + _N)
    t = math.sinh(math.atanh(math.sin(phi)) - c * math.atanh(c * math.sin(phi)))
    xi = math.atan2(t, math.cos(dlon))
    eta = math.atanh(math.sin(dlon) / math.sqrt(1.0 + t * t))
    east = eta
    north = xi
    for j, alpha in enumerate(_ALPHA, start=1):
        east += alpha * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
        north += alpha * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
    return 500000.0 + _K0 * _A * east, _K0 * _A * north, zone


class MgrsFrame:
    """Плоские координаты MGRS с фиксированным квадратом сетки, как в Autoware.

    x = восток UTM минус восток угла квадрата, y = север минус север угла,
    z = высота. За пределами квадрата координаты продолжаются непрерывно
    (x может быть больше 100 км), так заданы карты организаторов.
    По умолчанию зона 37 и квадрат с углом 300 000 / 6 100 000 м.
    """

    def __init__(self, zone=37, origin_east=300000.0, origin_north=6100000.0):
        self.zone = int(zone)
        self.origin = (float(origin_east), float(origin_north))

    def forward(self, lat, lon, alt=0.0):
        east, north, _ = utm_forward(lat, lon, self.zone)
        return east - self.origin[0], north - self.origin[1], alt

    def describe(self):
        return {'type': 'MGRS', 'utm_zone': self.zone, 'grid_origin': list(self.origin)}
