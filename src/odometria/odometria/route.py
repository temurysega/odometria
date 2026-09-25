"""Static path projection for relative ENU position after GNSS initialization.

Only the initial GNSS fix and receiver baseline are used online. The route is
an offline asset built from a separate training recording.
"""
from bisect import bisect_left
import json
import math
from pathlib import Path


EARTH_RADIUS = 6371000.0


class Route:
    def __init__(self, geographic_points, source=None):
        if len(geographic_points) < 2:
            raise ValueError('route needs at least two points')
        self.lat0, self.lon0 = geographic_points[0][:2]
        self.source = source
        self.cos_lat = math.cos(math.radians(self.lat0))
        self.points = [self.metric(*p) for p in geographic_points]
        self.distance = [0.0]
        for start, end in zip(self.points, self.points[1:]):
            self.distance.append(self.distance[-1] + math.dist(start[:2], end[:2]))

    def metric(self, lat, lon, alt=0.0):
        return (EARTH_RADIUS * math.radians(lon - self.lon0) * self.cos_lat,
                EARTH_RADIUS * math.radians(lat - self.lat0), alt)

    def nearest(self, point):
        best = None
        px, py = point[:2]
        for index, (start, end) in enumerate(zip(self.points, self.points[1:])):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length2 = dx * dx + dy * dy
            if length2 < 1e-8:
                continue
            fraction = max(0.0, min(1.0,
                ((px - start[0]) * dx + (py - start[1]) * dy) / length2))
            x, y = start[0] + fraction * dx, start[1] + fraction * dy
            error2 = (x - px) ** 2 + (y - py) ** 2
            if best is None or error2 < best[0]:
                length = math.sqrt(length2)
                best = (error2, self.distance[index] + fraction * length,
                        dx / length, dy / length)
        return best

    def at(self, distance):
        # Extrapolate beyond route ends so the output stays continuous.
        index = max(0, min(len(self.points) - 2,
                           bisect_left(self.distance, distance) - 1))
        start, end = self.points[index:index + 2]
        span = self.distance[index + 1] - self.distance[index]
        fraction = (distance - self.distance[index]) / span if span else 0.0
        return tuple(a + fraction * (b - a) for a, b in zip(start, end))


class Projector:
    def __init__(self, routes=None):
        self.routes = ([routes] if isinstance(routes, Route) else list(routes or []))
        self.route = self.routes[0] if self.routes else None
        self.anchor = None
        self.start_s = None
        self.direction = 1
        self.heading = (1.0, 0.0)
        self.map_error = None

    def initialize(self, lat, lon, alt, heading_e=None, heading_n=None,
                   max_map_error=100.0):
        if not all(math.isfinite(v) for v in (lat, lon, alt)):
            return False
        heading_norm = math.hypot(heading_e or 0.0, heading_n or 0.0)
        if heading_norm >= 2.0:
            self.heading = (heading_e / heading_norm, heading_n / heading_norm)
        if not self.routes:
            self.anchor = (lat, lon, alt)
            return True
        choices = []
        for route in self.routes:
            point = route.metric(lat, lon, alt)
            nearest = route.nearest(point)
            if nearest:
                direction = (1 if heading_norm < 2.0 or
                             nearest[2] * self.heading[0] +
                             nearest[3] * self.heading[1] >= 0 else -1)
                remaining = (route.distance[-1] - nearest[1] if direction == 1
                             else nearest[1])
                # A route ending at the start fix is a poor choice when the
                # observed heading points beyond its endpoint.
                score = nearest[0] + (200.0 ** 2 if remaining < 200 else 0.0)
                choices.append((score, route, point, nearest, direction))
        if not choices:
            return False
        _, self.route, self.anchor, nearest, self.direction = min(
            choices, key=lambda x: x[0])
        self.map_error = math.sqrt(nearest[0]) if nearest else None
        if nearest and self.map_error <= max_map_error:
            self.start_s = nearest[1]
        return True

    def position(self, distance):
        if self.start_s is not None:
            point = self.route.at(self.start_s + self.direction * distance)
            origin = self.route.at(self.start_s)
            return tuple(a - b for a, b in zip(point, origin))
        return (distance * self.heading[0], distance * self.heading[1], 0.0)


def load_routes():
    files = [Path(__file__).with_name(name) for name in
             ('route.json', 'route_reverse.json')]
    routes = []
    for path in files:
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            routes.append(Route(data['points'], data.get('source_bag')))
    return routes
