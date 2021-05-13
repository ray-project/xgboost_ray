from typing import Tuple

import unittest

import numpy as np
import xgboost as xgb
from xgboost_ray.compat import TrainingCallback

import ray

from xgboost_ray import RayDMatrix, train, RayParams

# From XGBoost documentation:
# https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
from xgboost_ray.session import get_actor_rank, put_queue


def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)


def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) / np.power(predt + 1, 2))


def squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return "PyRMSLE", float(np.sqrt(np.sum(elements) / len(y)))


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
            "objective": "binary:logistic",
            "seed": 1000
        }

        self.kwargs = {}

    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

    def _init_ray(self):
        if not ray.is_initialized():
            ray.init(num_cpus=4)

    def testCustomObjectiveFunction(self):
        """Ensure that custom objective functions work.

        Runs a custom objective function with pure XGBoost and
        XGBoost on Ray and compares the prediction outputs."""
        self._init_ray()

        params = self.params.copy()
        params.pop("objective", None)

        bst_xgb = xgb.train(
            params, xgb.DMatrix(self.x, self.y), obj=squared_log)

        bst_ray = train(
            params,
            RayDMatrix(self.x, self.y),
            ray_params=RayParams(num_actors=2),
            obj=squared_log,
            **self.kwargs)

        x_mat = xgb.DMatrix(self.x)
        pred_y_xgb = np.round(bst_xgb.predict(x_mat))
        pred_y_ray = np.round(bst_ray.predict(x_mat))

        self.assertSequenceEqual(list(pred_y_xgb), list(pred_y_ray))
        self.assertSequenceEqual(list(self.y), list(pred_y_ray))

    def testCustomMetricFunction(self):
        """Ensure that custom objective functions work.

        Runs a custom objective function with pure XGBoost and
        XGBoost on Ray and compares the prediction outputs."""
        self._init_ray()

        params = self.params.copy()
        params.pop("objective", None)
        params["disable_default_eval_metric"] = 1

        dtrain_xgb = xgb.DMatrix(self.x, self.y)
        evals_result_xgb = {}
        bst_xgb = xgb.train(
            params,
            dtrain_xgb,
            obj=squared_log,
            feval=rmsle,
            evals=[(dtrain_xgb, "dtrain")],
            evals_result=evals_result_xgb)

        dtrain_ray = RayDMatrix(self.x, self.y)
        evals_result_ray = {}
        bst_ray = train(
            params,
            dtrain_ray,
            ray_params=RayParams(num_actors=2),
            obj=squared_log,
            feval=rmsle,
            evals=[(dtrain_ray, "dtrain")],
            evals_result=evals_result_ray,
            **self.kwargs)

        x_mat = xgb.DMatrix(self.x)
        pred_y_xgb = np.round(bst_xgb.predict(x_mat))
        pred_y_ray = np.round(bst_ray.predict(x_mat))

        self.assertSequenceEqual(list(pred_y_xgb), list(pred_y_ray))
        self.assertSequenceEqual(list(self.y), list(pred_y_ray))

        self.assertTrue(
            np.allclose(
                evals_result_xgb["dtrain"]["PyRMSLE"],
                evals_result_ray["dtrain"]["PyRMSLE"],
                atol=0.1))

    def testCallbacks(self):
        class _Callback(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                print(f"My rank: {get_actor_rank()}")
                put_queue(("rank", get_actor_rank()))

        callback = _Callback()

        additional_results = {}
        train(
            self.params,
            RayDMatrix(self.x, self.y),
            ray_params=RayParams(num_actors=2),
            callbacks=[callback],
            additional_results=additional_results,
            **self.kwargs)

        self.assertEqual(len(additional_results["callback_returns"]), 2)
        self.assertTrue(
            all(rank == 0
                for (_, rank) in additional_results["callback_returns"][0]))
        self.assertTrue(
            all(rank == 1
                for (_, rank) in additional_results["callback_returns"][1]))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
