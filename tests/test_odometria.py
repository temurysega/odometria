"""Модульные тесты ядра одометрии без ROS: python -m unittest discover -s tests"""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'odometria'))

from odometria.core import CoreConfig, OdometryCore  # noqa: E402
from odometria.geodesy import LocalFrame, to_ecef  # noqa: E402
from odometria.observer import VelocityObserver  # noqa: E402
from odometria.track import Localizer, ParticleTrack, TrackMap  # noqa: E402
from odometria.traction import TractionModel  # noqa: E402

MODEL = TractionModel.load()
ORIGIN = (55.81, 37.46, 150.0)


def straight_map(length=3000, stops=()):
    """Прямой путь на восток от ORIGIN с ориентирами в точках stops."""
    points = [(float(x), 0.0, 0.0) for x in range(0, length + 1)]
    marks = [{'s': s, 'sigma': 0.35, 'count': 20, 'share': 0.8} for s in stops]
    return TrackMap(ORIGIN, points, marks)


def drive(observer, speeds, dt=0.1, command=0, start=100.0, rear=None):
    """Подаёт одинаковые показания тележек, rear позволяет исказить заднюю."""
    t = start
    for i, v in enumerate(speeds):
        observer.set_command(command)
        observer.update('front', t, v * 3.6)
        r = v if rear is None else rear(i, v)
        observer.update('rear', t + 0.02, None if r is None else r * 3.6)
        t += dt
    return t


class GeodesyTests(unittest.TestCase):
    def test_roundtrip(self):
        frame = LocalFrame(*ORIGIN)
        e, n, u = frame.from_geodetic(55.80, 37.40, 160.0)
        back = frame.from_ecef(*frame.to_ecef(e, n, u))
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip((e, n, u), back)))

    def test_ellipsoid_east_distance(self):
        # на широте 55,81° градус долготы около 62,7 км, сфера 6371 км даёт на 0,34 % меньше
        frame = LocalFrame(*ORIGIN)
        e, _, _ = frame.from_geodetic(55.81, 37.46 + 0.01, 150.0)
        self.assertAlmostEqual(e, 627.3, delta=0.5)
        self.assertEqual(len(to_ecef(*ORIGIN)), 3)


class TractionTests(unittest.TestCase):
    def test_traction_and_brake_signs(self):
        self.assertGreater(MODEL.acceleration(1.0, 3.0, 0.0, 15), 0.3)
        self.assertLess(MODEL.acceleration(-1.0, 8.0, 0.0, -15), -0.3)

    def test_standstill_holds(self):
        self.assertEqual(MODEL.acceleration(-0.5, 0.0, 0.03, -8), 0.0)

    def test_grade_reduces_acceleration(self):
        flat = MODEL.acceleration(0.5, 5.0, 0.0, 8)
        uphill = MODEL.acceleration(0.5, 5.0, 0.03, 8)
        self.assertAlmostEqual(flat - uphill, 9.80665 * 0.03, places=6)


class ObserverTests(unittest.TestCase):
    def test_tracks_clean_wheels(self):
        obs = VelocityObserver(MODEL)
        drive(obs, [5.0 + 0.05 * i for i in range(50)], command=5)
        self.assertAlmostEqual(obs.v, 5.0 + 0.05 * 49, delta=0.05)
        self.assertEqual(obs.health(obs.t), 'норма')

    def test_single_bogie_slide_is_rejected(self):
        obs = VelocityObserver(MODEL)
        speeds = [8.0 - 0.07 * i for i in range(40)]
        drive(obs, speeds, command=-5,
              rear=lambda i, v: v - (0.0 if i < 20 else min(3.0, 0.4 * (i - 19))))
        self.assertAlmostEqual(obs.v, speeds[-1], delta=0.3)
        self.assertTrue(obs.bogies['rear'].slipping)

    def test_common_spin_is_rejected(self):
        obs = VelocityObserver(MODEL)
        speeds = [3.0 + 0.03 * i for i in range(40)]
        wheels = [v + (0.0 if i < 20 else 0.35 * (i - 19)) for i, v in enumerate(speeds)]
        drive(obs, wheels, command=8,
              rear=lambda i, v: speeds[i] + (0.0 if i < 20 else 0.45 * (i - 19)))
        # оценка ближе к истинной скорости, чем к раскрученным колёсам
        self.assertLess(abs(obs.v - speeds[-1]), 0.25 * abs(wheels[-1] - speeds[-1]))
        self.assertTrue(obs.slip_detected)

    def test_emergency_brake_with_agreeing_bogies_is_accepted(self):
        obs = VelocityObserver(MODEL)
        speeds = [8.0] * 10 + [max(0.0, 8.0 - 0.45 * i) for i in range(1, 20)]
        drive(obs, speeds, command=0)
        self.assertLess(abs(obs.v - speeds[-1]), 0.3)

    def test_stuck_bogie_and_invalid_values(self):
        obs = VelocityObserver(MODEL)
        speeds = [6.0] * 30
        drive(obs, speeds, command=0, rear=lambda i, v: v if i < 10 else 0.0)
        self.assertAlmostEqual(obs.v, 6.0, delta=0.2)
        self.assertEqual(obs.update('front', obs.t + 0.1, float('nan')), 'некорректное значение')
        self.assertEqual(obs.update('front', obs.t + 0.1, 1e6), 'вне диапазона')
        self.assertTrue(math.isfinite(obs.v))

    def test_dropout_uses_model(self):
        obs = VelocityObserver(MODEL)
        t = drive(obs, [5.0] * 20, command=0)
        obs.predict(t + 1.5)
        self.assertTrue(0.0 <= obs.v < 5.5)
        self.assertGreater(obs.sigma_v, 0.1)


class TrackTests(unittest.TestCase):
    def test_closed_map_wraps(self):
        square = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, 100.0, 0.0), (0.0, 100.0, 0.0)]
        m = TrackMap(ORIGIN, square, closed=True)
        self.assertAlmostEqual(m.length, 400.0)
        self.assertAlmostEqual(m.point(410.0)[0], 10.0)
        self.assertAlmostEqual(m.delta(5.0, 395.0), 10.0)

    def test_projection_beyond_map_end(self):
        m = straight_map(100)
        cands = m.candidates(-20.0, 0.5, 30.0)
        self.assertAlmostEqual(min(c[1] for c in cands), -20.0, places=6)

    def test_particles_learn_scale_from_stops(self):
        # колёса занижают путь на 1,2 %, ориентиры каждые 400 м
        stops = list(range(400, 3000, 400))
        m = straight_map(3000, stops)
        f = ParticleTrack(0.0, sigma_s=0.3, sigma_k=0.008, count=800)
        loc = Localizer(m)
        loc.filter, loc.start_s = f, 0.0
        true_scale = 1.012
        travelled = 0.0
        for stop in stops:
            ds = stop / true_scale - travelled
            f.advance(ds)
            travelled += ds
            loc.last_landmark_distance = -1e9
            loc.try_landmark(travelled)
        self.assertAlmostEqual(f.k, true_scale, delta=0.003)
        self.assertAlmostEqual(f.s, stops[-1], delta=1.0)


class CoreTests(unittest.TestCase):
    def fix(self, core, t, east=0.0):
        frame = LocalFrame(*ORIGIN)
        lat, lon, alt = self._geodetic(frame, east)
        core.on_fix('master', t, lat, lon, alt, 2)
        lat2, lon2, alt2 = self._geodetic(frame, east + 12.4)
        core.on_fix('rover', t, lat2, lon2, alt2, 2)

    @staticmethod
    def _geodetic(frame, east):
        # малый сдвиг на восток вдоль параллели
        lat = ORIGIN[0]
        lon = ORIGIN[1] + east / (111320.0 * math.cos(math.radians(lat)))
        return lat, lon, ORIGIN[2]

    def run_core(self, core, seconds=30.0, speed=5.0, t0=1000.0, fixes=True):
        outs = []
        t = t0
        core.on_wheel('front', t0 - 0.1, speed * 3.6)
        while t < t0 + seconds:
            if fixes and t < t0 + 3.0:
                self.fix(core, t, east=speed * (t - t0))
            outs += core.on_command(t + 0.05, 5)
            outs += core.on_wheel('front', t, speed * 3.6)
            outs += core.on_wheel('rear', t + 0.03, speed * 3.6)
            t += 0.1
        return outs

    def test_outputs_on_grid_and_rate(self):
        core = OdometryCore(CoreConfig(), track_map=straight_map(1000))
        outs = self.run_core(core)
        stamps = [o.stamp for o in outs]
        self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])))
        self.assertTrue(all(abs(s * 50 - round(s * 50)) < 1e-6 for s in stamps))
        self.assertGreater(len(outs) / 30.0, 45.0)

    def test_gnss_ignored_after_init(self):
        core = OdometryCore(CoreConfig(), track_map=straight_map(1000))
        self.run_core(core, seconds=5.0)
        self.assertFalse(core.gnss_needed)
        self.assertFalse(core.on_fix('master', 2000.0, 0.0, 0.0, 0.0, 2))

    def test_position_follows_track(self):
        core = OdometryCore(CoreConfig(), track_map=straight_map(1000))
        outs = self.run_core(core, seconds=20.0, speed=5.0)
        x, y, _ = outs[-1].position
        self.assertAlmostEqual(y, 0.0, delta=0.5)
        self.assertAlmostEqual(x, 5.0 * (outs[-1].stamp - 1000.0), delta=1.5)

    def test_bad_inputs_do_not_crash(self):
        core = OdometryCore(CoreConfig(), track_map=None)
        for bad in (float('nan'), None, 'x', float('inf')):
            core.on_wheel('front', bad, 10.0)
            core.on_wheel('front', 1000.0, bad)
            core.on_command(bad, 3)
        core.on_command(1000.0, 99)
        core.on_command(1000.0, 2.5)
        self.assertGreater(core.counters.rejected_input, 0)

    def test_reset_on_new_recording(self):
        core = OdometryCore(CoreConfig(), track_map=None)
        self.run_core(core, seconds=5.0, fixes=False)
        self.run_core(core, seconds=5.0, t0=500.0, fixes=False)
        self.assertEqual(core.counters.resets, 1)

    def test_relative_odometry_without_gnss(self):
        core = OdometryCore(CoreConfig(), track_map=straight_map(1000))
        outs = self.run_core(core, seconds=10.0, fixes=False)
        self.assertGreater(outs[-1].position[0], 40.0)
        self.assertTrue(core.gnss_needed)


if __name__ == '__main__':
    unittest.main()
