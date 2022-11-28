import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ray
from ray import tune
from ray.tune import TuneError
from ray.tune.integration.xgboost import \
    TuneReportCallback as OrigTuneReportCallback, \
    TuneReportCheckpointCallback as OrigTuneReportCheckpointCallback

from xgboost_ray import RayDMatrix, train, RayParams
from xgboost_ray.tune import TuneReportCallback,\
    TuneReportCheckpointCallback, _try_add_tune_callback

try:
    from ray.air import Checkpoint
except Exception:

    class Checkpoint:
        pass


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

        def train_func(ray_params,
                       callbacks=None,
                       check_for_spread_strategy=False,
                       **kwargs):
            def _inner_train(config, checkpoint_dir):
                if check_for_spread_strategy:
                    assert tune.get_trial_resources().strategy == "SPREAD"
                train_set = RayDMatrix(x, y)
                train(
                    config["xgb"],
                    dtrain=train_set,
                    ray_params=ray_params,
                    num_boost_round=config["num_boost_round"],
                    evals=[(train_set, "train")],
                    callbacks=callbacks,
                    **kwargs)

            return _inner_train

        self.train_func = train_func
        self.experiment_dir = tempfile.mkdtemp()

    def tearDown(self):
        ray.shutdown()
        shutil.rmtree(self.experiment_dir)

    # noinspection PyTypeChecker
    @patch.dict(os.environ, {"TUNE_RESULT_DELIM": "/"})
    def testNumIters(self):
        """Test that the number of reported tune results is correct"""
        ray_params = RayParams(cpus_per_actor=1, num_actors=2)
        params = self.params.copy()
        params["num_boost_round"] = tune.grid_search([1, 3])
        analysis = tune.run(
            self.train_func(ray_params),
            config=self.params,
            resources_per_trial=ray_params.get_tune_resources(),
            num_samples=1)

        self.assertSequenceEqual(
            list(analysis.results_df["training_iteration"]),
            list(analysis.results_df["config/num_boost_round"]))

    def testNumItersClient(self):
        """Test ray client mode"""
        if ray.__version__ <= "1.2.0":
            self.skipTest("Ray client mocks do not work in Ray <= 1.2.0")

        from ray.util.client.ray_client_helpers import ray_start_client_server

        self.assertFalse(ray.util.client.ray.is_connected())
        with ray_start_client_server():
            self.assertTrue(ray.util.client.ray.is_connected())
            self.testNumIters()

    def testPlacementOptions(self):
        ray_params = RayParams(
            cpus_per_actor=1,
            num_actors=1,
            placement_options={"strategy": "SPREAD"})
        tune.run(
            self.train_func(ray_params, check_for_spread_strategy=True),
            config=self.params,
            resources_per_trial=ray_params.get_tune_resources(),
            num_samples=1)

    def testElasticFails(self):
        """Test if error is thrown when using Tune with elastic training."""
        ray_params = RayParams(
            cpus_per_actor=1, num_actors=1, elastic_training=True)
        with self.assertRaises(TuneError):
            tune.run(
                self.train_func(ray_params),
                config=self.params,
                resources_per_trial=ray_params.get_tune_resources(),
                num_samples=1)

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
        ray_params = RayParams(cpus_per_actor=1, num_actors=2)
        analysis = tune.run(
            self.train_func(
                ray_params,
                callbacks=[TuneReportCheckpointCallback(frequency=1)]),
            config=self.params,
            resources_per_trial=ray_params.get_tune_resources(),
            num_samples=1,
            metric="train-mlogloss",
            mode="min",
            log_to_file=True,
            local_dir=self.experiment_dir)

        if isinstance(analysis.best_checkpoint, Checkpoint):
            self.assertTrue(analysis.best_checkpoint)
        else:
            self.assertTrue(os.path.exists(analysis.best_checkpoint))

    def testEndToEndCheckpointingOrigTune(self):
        ray_params = RayParams(cpus_per_actor=1, num_actors=2)
        analysis = tune.run(
            self.train_func(
                ray_params,
                callbacks=[OrigTuneReportCheckpointCallback(frequency=1)]),
            config=self.params,
            resources_per_trial=ray_params.get_tune_resources(),
            num_samples=1,
            metric="train-mlogloss",
            mode="min",
            log_to_file=True,
            local_dir=self.experiment_dir)

        if isinstance(analysis.best_checkpoint, Checkpoint):
            self.assertTrue(analysis.best_checkpoint)
        else:
            self.assertTrue(os.path.exists(analysis.best_checkpoint))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
