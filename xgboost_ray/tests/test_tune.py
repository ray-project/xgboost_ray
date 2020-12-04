import os
import unittest

import numpy as np

import ray
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback

from xgboost_ray import RayDMatrix, train, RayParams


class XGBoostRayTuneTest(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=8)
        repeat = 8  # Repeat data a couple of times for stability
        x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        y = np.array([0, 1, 2, 3] * repeat)

        self.params = {
            "xgb": {
                "booster": "gbtree",
                "nthread": 1,
                "max_depth": 2,
                "objective": "multi:softmax",
                "num_class": 4,
                "eval_metric": ["mlogloss", "merror"],
            },
            "num_boost_round": tune.choice([1, 3])
        }

        def train_func(config, checkpoint_dir=None, num_actors=1, **kwargs):
            train_set = RayDMatrix(x, y)
            train(
                config["xgb"],
                dtrain=train_set,
                ray_params=RayParams(cpus_per_actor=1, num_actors=num_actors),
                num_boost_round=config["num_boost_round"],
                evals=[(train_set, "train")],
                **kwargs)

        self.train_func = train_func

    def tearDown(self):
        ray.shutdown()

    # noinspection PyTypeChecker
    def testNumIters(self):
        analysis = tune.run(
            self.train_func,
            config=self.params,
            resources_per_trial={
                "cpu": 1,
                "extra_cpu": 1
            },
            num_samples=2)

        self.assertTrue(
            all(analysis.results_df["training_iteration"] ==
                analysis.results_df["config.num_boost_round"]))

    def testCheckpointing(self):
        analysis = tune.run(
            tune.with_parameters(
                self.train_func,
                num_actors=2,
                callbacks=[TuneReportCheckpointCallback(frequency=1)]),
            config=self.params,
            resources_per_trial={
                "cpu": 1,
                "extra_cpu": 1
            },
            num_samples=2,
            metric="train-mlogloss",
            mode="min")

        self.assertTrue(os.path.exists(analysis.best_checkpoint))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
