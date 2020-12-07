import os
import unittest
from unittest.mock import patch

import numpy as np

import ray
from ray import tune
from ray.tune.integration.xgboost import \
    TuneReportCallback as OrigTuneReportCallback, \
    TuneReportCheckpointCallback as OrigTuneReportCheckpointCallback

from xgboost_ray import RayDMatrix, train, RayParams
from xgboost_ray.tune import TuneReportCallback,\
    TuneReportCheckpointCallback, _try_add_tune_callback


class XGBoostRayTuneTest(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=4)
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

    def testReplaceTuneCheckpoints(self):
        """Test if ray.tune.integration.xgboost callbacks are replaced"""
        # Report callback
        in_cp = [OrigTuneReportCallback(metrics="met")]
        in_dict = {"callbacks": in_cp}

        with patch("xgboost_ray.tune.is_session_enabled") as mocked:
            mocked.return_value = True
            _try_add_tune_callback(in_dict)

        replaced = in_dict["callbacks"][0]
        self.assertTrue(isinstance(replaced, TuneReportCallback))
        self.assertSequenceEqual(replaced._metrics, ["met"])

        # Report and checkpointing callback
        in_cp = [
            OrigTuneReportCheckpointCallback(metrics="met", filename="test")
        ]
        in_dict = {"callbacks": in_cp}

        with patch("xgboost_ray.tune.is_session_enabled") as mocked:
            mocked.return_value = True
            _try_add_tune_callback(in_dict)

        replaced = in_dict["callbacks"][0]
        self.assertTrue(isinstance(replaced, TuneReportCheckpointCallback))
        self.assertSequenceEqual(replaced._report._metrics, ["met"])
        self.assertEqual(replaced._checkpoint._filename, "test")

    def testEndToEndCheckpointing(self):
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

    def testEndToEndCheckpointingOrigTune(self):
        analysis = tune.run(
            tune.with_parameters(
                self.train_func,
                num_actors=2,
                callbacks=[OrigTuneReportCheckpointCallback()]),
            config=self.params,
            resources_per_trial={
                "cpu": 1,
                "extra_cpu": 1
            },
            num_samples=2,
            metric="train-mlogloss",
            mode="min",
            log_to_file=True)

        self.assertTrue(os.path.exists(analysis.best_checkpoint))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
