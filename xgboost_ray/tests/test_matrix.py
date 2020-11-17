import numpy as np
import unittest

import ray

from xgboost_ray import RayDMatrix


class XGBoostRayDMatrixTest(unittest.TestCase):
    """This test suite validates core RayDMatrix functionality."""

    def setUp(self):
        repeat = 8  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        self.y = np.array([0, 1, 2, 3] * repeat)

    def testSameObject(self):
        """Test that matrices are recognized as the same in an actor task."""
        ray.init(num_cpus=1)

        @ray.remote
        def same(one, two):
            return one == two

        data = RayDMatrix(self.x, self.y)
        self.assertTrue(ray.get(same.remote(data, data)))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
