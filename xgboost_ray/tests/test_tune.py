import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import ray
from ray import tune

from xgboost_ray import RayDMatrix, train
from xgboost_ray.matrix import concat_dataframes


class XGBoostRayDMatrixTest(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=4)
        repeat = 8  # Repeat data a couple of times for stability
        x = np.array([
                              [1, 0, 0, 0],  # Feature 0 -> Label 0
                              [0, 1, 0, 0],  # Feature 1 -> Label 1
                              [0, 0, 1, 1],  # Feature 2+3 -> Label 2
                              [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
                          ] * repeat)
        y = np.array([
                              0, 1, 2, 3
                          ] * repeat)

        self.params = {
            "xgb": {
                "booster": "gbtree",
                "nthread": 1,
                "max_depth": 2,
                "objective": "multi:softmax",
                "num_class": 4,
            },
            "num_boost_round": tune.choice([1, 3])
        }

        def train_func(config):
            train_set = RayDMatrix(x, y)
            train(
                config["xgb"],
                dtrain=train_set,
                cpus_per_actor=1,
                num_actors=1,
                num_boost_round=config["num_boost_round"])

        self.train_func = train_func


    def tearDown(self):
        ray.shutdown()

    # noinspection PyTypeChecker
    def test_num_iters(self):
        analysis = tune.run(self.train_func, config=self.params,
                            resources_per_trial={"cpu": 1,"extra_cpu": 1},
                            num_samples=2)

        self.assertTrue(all(analysis.results_df["training_iteration"] ==
                            analysis.results_df["config.num_boost_round"]))

if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
