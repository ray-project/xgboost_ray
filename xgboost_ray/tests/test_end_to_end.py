import os
import shutil
import tempfile

import numpy as np
import unittest
import xgboost as xgb

import ray
from ray.exceptions import RayActorError, RayTaskError

from xgboost_ray import RayParams, train, RayDMatrix, predict, RayShardingMode
from xgboost_ray.main import RayXGBoostTrainingError
from xgboost_ray.callback import DistributedCallback
from xgboost_ray.tests.utils import get_num_trees


def _make_callback(tmpdir: str) -> DistributedCallback:
    class TestDistributedCallback(DistributedCallback):
        logdir = tmpdir

        def on_init(self, actor, *args, **kwargs):
            log_file = os.path.join(self.logdir, f"rank_{actor.rank}.log")
            actor.log_fp = open(log_file, "at")
            actor.log_fp.write(f"Actor {actor.rank}: Init\n")
            actor.log_fp.flush()

        def before_data_loading(self, actor, data, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: Before loading\n")
            actor.log_fp.flush()

        def after_data_loading(self, actor, data, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: After loading\n")
            actor.log_fp.flush()

        def before_train(self, actor, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: Before train\n")
            actor.log_fp.flush()

        def after_train(self, actor, result_dict, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: After train\n")
            actor.log_fp.flush()

        def before_predict(self, actor, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: Before predict\n")
            actor.log_fp.flush()

        def after_predict(self, actor, predictions, *args, **kwargs):
            actor.log_fp.write(f"Actor {actor.rank}: After predict\n")
            actor.log_fp.flush()

    return TestDistributedCallback()


class XGBoostRayEndToEndTest(unittest.TestCase):
    """In this test suite we validate Ray-XGBoost multi class prediction.

    First, we validate that XGBoost is able to achieve 100% accuracy on
    a simple training task.

    Then we split the dataset into two halves. These halves don't have access
    to all relevant data, so overfit on their respective data. I.e. the first
    half always predicts feature 2 -> label 2, while the second half always
    predicts feature 2 -> label 3.

    We then train using Ray XGBoost. Again both halves will be trained
    separately, but because of Rabit's allreduce, they should end up being
    able to achieve 100% accuracy, again."""

    def setUp(self):
        repeat = 8  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        self.y = np.array([0, 1, 2, 3] * repeat)

        self.params = {
            "booster": "gbtree",
            "nthread": 1,
            "max_depth": 2,
            "objective": "multi:softmax",
            "num_class": 4
        }

    def tearDown(self):
        if ray.is_initialized:
            ray.shutdown()

    def testSingleTraining(self):
        """Test that XGBoost learns to predict full matrix"""
        dtrain = xgb.DMatrix(self.x, self.y)
        bst = xgb.train(self.params, dtrain, num_boost_round=2)

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))

    def testHalfTraining(self):
        """Test that XGBoost learns to predict half matrices individually"""
        x_first = self.x[::2]
        y_first = self.y[::2]

        x_second = self.x[1::2]
        y_second = self.y[1::2]

        # Test case: The first model only sees feature 2 --> label 2
        # and the second model only sees feature 2 --> label 3
        test_X = xgb.DMatrix(np.array([[0, 0, 1, 1], [0, 0, 1, 0]]))
        test_y_first = [2, 2]
        test_y_second = [3, 3]

        # First half
        dtrain = xgb.DMatrix(x_first, y_first)
        bst = xgb.train(self.params, dtrain, num_boost_round=2)

        x_mat = xgb.DMatrix(x_first)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(y_first), list(pred_y))

        pred_test = bst.predict(test_X)
        self.assertSequenceEqual(test_y_first, list(pred_test))

        # Second half
        dtrain = xgb.DMatrix(x_second, y_second)
        bst = xgb.train(self.params, dtrain, num_boost_round=2)

        x_mat = xgb.DMatrix(x_second)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(y_second), list(pred_y))

        pred_test = bst.predict(test_X)
        self.assertSequenceEqual(test_y_second, list(pred_test))

    def _testJointTraining(self,
                           sharding=RayShardingMode.INTERLEAVED,
                           softprob=False):
        """Train with Ray. The data will be split, but the trees
        should be combined together and find the true model."""
        params = self.params.copy()
        if softprob:
            params["objective"] = "multi:softprob"

        bst = train(
            params,
            RayDMatrix(self.x, self.y, sharding=sharding),
            ray_params=RayParams(num_actors=2))

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        if softprob:
            pred_y = np.argmax(pred_y, axis=1)
        pred_y = pred_y.astype(int)
        self.assertSequenceEqual(list(self.y), list(pred_y))

        x_mat = RayDMatrix(self.x, sharding=sharding)
        pred_y = predict(bst, x_mat, ray_params=RayParams(num_actors=2))
        if softprob:
            pred_y = np.argmax(pred_y, axis=1)
        pred_y = pred_y.astype(int)
        self.assertSequenceEqual(list(self.y), list(pred_y))

        # try on an odd number of rows
        bst = train(
            params,
            RayDMatrix(self.x[:-1], self.y[:-1], sharding=sharding),
            ray_params=RayParams(num_actors=2))

        x_mat = RayDMatrix(self.x[:-1], sharding=sharding)
        pred_y = predict(bst, x_mat, ray_params=RayParams(num_actors=2))
        if softprob:
            pred_y = np.argmax(pred_y, axis=1)
        pred_y = pred_y.astype(int)
        self.assertSequenceEqual(list(self.y[:-1]), list(pred_y))

    def testJointTrainingInterleaved(self):
        ray.init(num_cpus=2, num_gpus=0)
        self._testJointTraining(sharding=RayShardingMode.INTERLEAVED)
        self._testJointTraining(
            sharding=RayShardingMode.INTERLEAVED, softprob=True)

    def testJointTrainingBatch(self):
        ray.init(num_cpus=2, num_gpus=0)
        self._testJointTraining(sharding=RayShardingMode.BATCH)
        self._testJointTraining(sharding=RayShardingMode.BATCH, softprob=True)

    def testTrainPredict(self,
                         init=True,
                         remote=None,
                         softprob=False,
                         **ray_param_dict):
        """Train with evaluation and predict"""
        if init:
            ray.init(num_cpus=2, num_gpus=0)

        dtrain = RayDMatrix(self.x, self.y)

        params = self.params
        if softprob:
            params = params.copy()
            params["objective"] = "multi:softprob"

        evals_result = {}
        bst = train(
            params,
            dtrain,
            num_boost_round=38,
            ray_params=RayParams(num_actors=2, **ray_param_dict),
            evals=[(dtrain, "dtrain")],
            evals_result=evals_result,
            _remote=remote)

        self.assertEqual(get_num_trees(bst), 38)

        self.assertTrue("dtrain" in evals_result)

        x_mat = RayDMatrix(self.x)
        pred_y = predict(
            bst,
            x_mat,
            ray_params=RayParams(num_actors=2, **ray_param_dict),
            _remote=remote)

        if softprob:
            self.assertEqual(pred_y.shape[1], len(np.unique(self.y)))
            pred_y = np.argmax(pred_y, axis=1)

        self.assertSequenceEqual(list(self.y), list(pred_y))

    def testTrainPredictSoftprob(self):
        """Train with evaluation and predict on softprob objective
        (which returns predictions in a 2d array)
        """
        self.testTrainPredict(init=True, softprob=True)

    def testTrainPredictRemote(self):
        """Train with evaluation and predict in a remote call"""
        self.testTrainPredict(init=True, remote=True)

    def testTrainPredictClient(self):
        """Train with evaluation and predict in a client session"""
        if ray.__version__ <= "1.2.0":
            self.skipTest("Ray client mocks do not work in Ray <= 1.2.0")
        from ray.util.client.ray_client_helpers import ray_start_client_server

        ray.init(num_cpus=2, num_gpus=0)
        self.assertFalse(ray.util.client.ray.is_connected())
        with ray_start_client_server():
            self.assertTrue(ray.util.client.ray.is_connected())

            self.testTrainPredict(init=False, remote=None)

    def testDistributedCallbacksTrainPredict(self, init=True, remote=False):
        """Test distributed callbacks for train/predict"""
        tmpdir = tempfile.mkdtemp()
        test_callback = _make_callback(tmpdir)

        self.testTrainPredict(
            init=init, remote=remote, distributed_callbacks=[test_callback])
        rank_0_log_file = os.path.join(tmpdir, "rank_0.log")
        rank_1_log_file = os.path.join(tmpdir, "rank_1.log")
        self.assertTrue(os.path.exists(rank_1_log_file))

        rank_0_log = open(rank_0_log_file, "rt").read()
        self.assertEqual(
            rank_0_log, "Actor 0: Init\n"
            "Actor 0: Before loading\n"
            "Actor 0: After loading\n"
            "Actor 0: Before train\n"
            "Actor 0: After train\n"
            "Actor 0: Init\n"
            "Actor 0: Before loading\n"
            "Actor 0: After loading\n"
            "Actor 0: Before predict\n"
            "Actor 0: After predict\n")
        shutil.rmtree(tmpdir)

    def testDistributedCallbacksTrainPredictClient(self):
        """Test distributed callbacks for train/predict via Ray client"""

        if ray.__version__ <= "1.2.0":
            self.skipTest("Ray client mocks do not work in Ray <= 1.2.0")
        from ray.util.client.ray_client_helpers import ray_start_client_server

        ray.init(num_cpus=2, num_gpus=0)
        self.assertFalse(ray.util.client.ray.is_connected())
        with ray_start_client_server():
            self.assertTrue(ray.util.client.ray.is_connected())

            self.testDistributedCallbacksTrainPredict(init=False, remote=None)

    def testFailPrintErrors(self):
        """Test that XGBoost training errors are propagated"""
        x = np.random.uniform(0, 1, size=(100, 4))
        y = np.random.randint(0, 2, size=100)

        train_set = RayDMatrix(x, y)

        try:
            train(
                {
                    "objective": "multi:softmax",
                    "num_class": 2,
                    "eval_metric": ["logloss", "error"]
                },  # This will error
                train_set,
                evals=[(train_set, "train")],
                ray_params=RayParams(num_actors=1, max_actor_restarts=0))
        except RuntimeError as exc:
            self.assertTrue(exc.__cause__)
            self.assertTrue(isinstance(exc.__cause__, RayActorError))

            self.assertTrue(exc.__cause__.__cause__)
            self.assertTrue(isinstance(exc.__cause__.__cause__, RayTaskError))

            self.assertTrue(exc.__cause__.__cause__.cause)
            self.assertTrue(
                isinstance(exc.__cause__.__cause__.cause,
                           RayXGBoostTrainingError))

            self.assertIn("label and prediction size not match",
                          str(exc.__cause__.__cause__))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
