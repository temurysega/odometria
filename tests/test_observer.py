import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'odometria'))
from odometria.estimator import Config, Observer


class ObserverTests(unittest.TestCase):
    def test_recorded_wheel_units_and_initial_motion(self):
        observer = Observer()
        estimate = observer.step(0.0, 0, 36.0, 36.0)
        self.assertEqual(estimate.trusted_bogies, 2)
        self.assertAlmostEqual(estimate.velocity, 10.0, places=2)

    def test_common_wheel_slip_is_rejected(self):
        observer = Observer()
        observer.step(0.0, 0, 36.0, 36.0)
        estimate = observer.step(0.1, 0, 72.0, 72.0)
        self.assertEqual(estimate.status, 'suspect_wheels')
        self.assertLess(estimate.velocity, 11.0)

    def test_one_bogie_disagreement_uses_model(self):
        observer = Observer()
        observer.step(0.0, 0, 18.0, 18.0)
        estimate = observer.step(0.1, 0, 18.0, 27.0)
        self.assertEqual(estimate.trusted_bogies, 0)
        self.assertLess(estimate.velocity, 6.0)

    def test_dropout_keeps_position_and_uncertainty_finite(self):
        observer = Observer()
        first = observer.step(0.0, 0, 18.0, 18.0)
        second = observer.step(0.2, 0, None, None)
        self.assertGreater(second.distance, first.distance)
        self.assertGreater(second.sigma_distance, first.sigma_distance)
        self.assertTrue(math.isfinite(second.velocity))

    def test_invalid_timestamp_and_controller(self):
        observer = Observer()
        observer.step(0.0, 0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            observer.step(0.0, 0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            observer.step(0.1, 16, 0.0, 0.0)

    def test_long_gap_restarts_relative_trajectory(self):
        observer = Observer(Config(max_gap=1.0))
        observer.step(0.0, 0, 18.0, 18.0)
        observer.step(0.2, 0, 18.0, 18.0)
        estimate = observer.step(3.0, 0, 0.0, 0.0)
        self.assertAlmostEqual(estimate.distance, 0.0)


if __name__ == '__main__':
    unittest.main()
