from typing import Tuple

import unittest

import numpy as np
import xgboost as xgb

import ray

from xgboost_ray import RayDMatrix, train


class XGBoostAPITest(unittest.TestCase):
    """This test suite validates core XGBoost API functionality."""

    def setUp(self):
        repeat = 8  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 0
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 1
        ] * repeat)
        self.y = np.array([0, 1, 0, 1] * repeat)

        self.params = {
            "booster": "gbtree",
            "tree_method": "hist",
            "nthread": 1,
            "max_depth": 2,
            "objective": "binary:logloss",
            "seed": 1000
        }

    def testCustomObjectiveFunction(self):
        """Ensure that custom objective functions work.

        Runs a custom objective function with pure XGBoost and
        XGBoost on Ray and compares the prediction outputs."""
        ray.init(num_cpus=4)

        params = self.params.copy()
        params.pop("objective", None)

        # From XGBoost documentation
        def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
            y = dtrain.get_label()
            return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

        def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
            y = dtrain.get_label()
            return (
                (-np.log1p(predt) + np.log1p(y) + 1) / np.power(predt + 1, 2))

        def squared_log(predt: np.ndarray,
                        dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
            predt[predt < -1] = -1 + 1e-6
            grad = gradient(predt, dtrain)
            hess = hessian(predt, dtrain)
            return grad, hess

        bst_xgb = xgb.train(
            params, xgb.DMatrix(self.x, self.y), obj=squared_log)

        bst_ray = train(
            params, RayDMatrix(self.x, self.y), num_actors=2, obj=squared_log)

        x_mat = xgb.DMatrix(self.x)
        pred_y_xgb = np.round(bst_xgb.predict(x_mat))
        pred_y_ray = np.round(bst_ray.predict(x_mat))

        self.assertSequenceEqual(list(pred_y_xgb), list(pred_y_ray))
        self.assertSequenceEqual(list(self.y), list(pred_y_ray))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
