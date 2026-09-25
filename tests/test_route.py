import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'odometria'))
from odometria.route import Projector, Route


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.route = Route([(55.0, 37.0, 100.0),
                            (55.0, 37.001, 101.0),
                            (55.0, 37.002, 102.0)])

    def test_forward_and_reverse_coordinates(self):
        forward = Projector(self.route)
        forward.initialize(55.0, 37.0, 100.0, 10.0, 0.0)
        self.assertAlmostEqual(forward.position(40)[0], 40, places=2)
        self.assertAlmostEqual(forward.position(0)[0], 0, places=5)
        reverse = Projector(self.route)
        reverse.initialize(55.0, 37.002, 102.0, -10.0, 0.0)
        self.assertEqual(reverse.direction, -1)
        self.assertAlmostEqual(reverse.position(40)[0], -40, places=2)

    def test_off_map_uses_initial_heading(self):
        projector = Projector(self.route)
        projector.initialize(55.01, 37.0, 100.0, 0.0, 12.0)
        self.assertIsNone(projector.start_s)
        self.assertAlmostEqual(projector.position(10)[0], 0, places=5)
        self.assertAlmostEqual(projector.position(10)[1], 10, places=5)

    def test_selects_nearest_route(self):
        other = Route([(55.01, 37.0, 100.0), (55.01, 37.001, 100.0)])
        projector = Projector([other, self.route])
        projector.initialize(55.0, 37.0, 100.0, 10.0, 0.0)
        self.assertIs(projector.route, self.route)
        self.assertTrue(math.isclose(projector.position(10)[0], 10, abs_tol=1e-5))

    def test_heading_without_map(self):
        projector = Projector([])
        self.assertTrue(projector.initialize(55.0, 37.0, 100.0, 0.0, 10.0))
        self.assertEqual(projector.position(5), (0.0, 5.0, 0.0))


if __name__ == '__main__':
    unittest.main()
