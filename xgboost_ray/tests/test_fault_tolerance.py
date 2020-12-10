import json
import os
import shutil
import tempfile
import time

import numpy as np
import unittest
import xgboost as xgb
from xgboost.callback import TrainingCallback

import ray

from xgboost_ray import train, RayDMatrix, RayParams
from xgboost_ray.session import put_queue, get_actor_rank
from xgboost_ray.tests.utils import flatten_obj


def tree_obj(bst: xgb.Booster):
    return [json.loads(j) for j in bst.get_dump(dump_format="json")]


def _kill_callback(die_lock_file: str,
                   actor_rank: int = 0,
                   fail_iteration: int = 6):
    class _KillCallback(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if get_actor_rank() == actor_rank:
                put_queue((epoch, time.time()))
            if get_actor_rank() == actor_rank and \
                    epoch == fail_iteration and \
                    not os.path.exists(die_lock_file):
                # Only die once
                if os.path.exists(die_lock_file):
                    return

                # Get PID
                pid = os.getpid()
                print(f"Killing process: {pid}")
                with open(die_lock_file, "wt") as fp:
                    fp.write("")

                time.sleep(2)
                os.kill(pid, 9)

    return _KillCallback()


def _fail_callback(die_lock_file: str,
                   actor_rank: int = 0,
                   fail_iteration: int = 6):
    class _FailCallback(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):

            if get_actor_rank() == actor_rank:
                put_queue((epoch, time.time()))
            if get_actor_rank() == actor_rank and \
               epoch == fail_iteration and \
               not os.path.exists(die_lock_file):
                # Only die once
                if os.path.exists(die_lock_file):
                    return

                with open(die_lock_file, "wt") as fp:
                    fp.write("")
                time.sleep(2)
                import sys
                sys.exit(1)

    return _FailCallback()


def _checkpoint_callback(frequency: int = 1, before_iteration=False):
    class _CheckpointCallback(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch % frequency == 0:
                put_queue(model.save_raw())

    if before_iteration:
        _CheckpointCallback.before_iteration = \
            _CheckpointCallback.after_iteration

    return _CheckpointCallback()


class XGBoostRayFaultToleranceTest(unittest.TestCase):
    """In this test suite we validate fault tolerance when a Ray actor dies.

    For this, we set up a callback that makes one worker die exactly once.
    """

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
            "eval_metric": "mlogloss",
            "num_class": 4
        }

        self.tmpdir = str(tempfile.mkdtemp())

        self.die_lock_file = "/tmp/died_worker.lock"
        if os.path.exists(self.die_lock_file):
            os.remove(self.die_lock_file)

        ray.init(num_cpus=2, num_gpus=0, log_to_driver=True)

    def tearDown(self) -> None:
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        ray.shutdown()

    def testTrainingContinuationKilled(self):
        """This should continue after one actor died."""
        actors = [None, None]
        additional_results = {}

        bst = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[_kill_callback(self.die_lock_file)],
            num_boost_round=20,
            ray_params=RayParams(max_actor_restarts=1, num_actors=2),
            additional_results=additional_results,
            _actors=actors)

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

        # End with two working actors
        self.assertTrue(actors[0])
        self.assertTrue(actors[1])

        # Two workers finished, so N=32
        self.assertEqual(additional_results["total_n"], 32)

    def testTrainingContinuationElasticKilled(self):
        """This should continue after one actor died."""
        actors = [None, None]
        additional_results = {}

        bst = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[_kill_callback(self.die_lock_file)],
            num_boost_round=20,
            ray_params=RayParams(
                max_actor_restarts=1,
                num_actors=2,
                elastic_training=True,
                max_failed_actors=1),
            additional_results=additional_results,
            _actors=actors)

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

        # First actor does not get recreated
        self.assertEqual(actors[0], None)
        self.assertTrue(actors[1])

        # Only one worker finished, so n=16
        self.assertEqual(additional_results["total_n"], 16)

    def testTrainingContinuationElasticFailed(self):
        """This should continue after one actor failed training."""
        actors = [None, None]
        additional_results = {}

        bst = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[_fail_callback(self.die_lock_file)],
            num_boost_round=20,
            ray_params=RayParams(
                max_actor_restarts=1,
                num_actors=2,
                elastic_training=True,
                max_failed_actors=1),
            additional_results=additional_results,
            _actors=actors)

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

        # End with two working actors since only the training failed
        self.assertTrue(actors[0])
        self.assertTrue(actors[1])

        # Two workers finished, so n=32
        self.assertEqual(additional_results["total_n"], 32)

    def testTrainingStop(self):
        """This should now stop training after one actor died."""
        # The `train()` function raises a RuntimeError
        with self.assertRaises(RuntimeError):
            train(
                self.params,
                RayDMatrix(self.x, self.y),
                callbacks=[_fail_callback(self.die_lock_file)],
                num_boost_round=20,
                ray_params=RayParams(max_actor_restarts=0, num_actors=2))

    def testCheckpointContinuationValidity(self):
        """Test that checkpoints are stored and loaded correctly"""

        # Train once, get checkpoint via callback returns
        res_1 = {}
        bst_1 = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[
                _checkpoint_callback(frequency=1, before_iteration=False)
            ],
            num_boost_round=2,
            ray_params=RayParams(num_actors=2),
            additional_results=res_1)
        last_checkpoint_1 = res_1["callback_returns"][0][-1]
        last_checkpoint_other_rank_1 = res_1["callback_returns"][1][-1]

        # Sanity check
        lc1 = xgb.Booster()
        lc1.load_model(last_checkpoint_1)
        self.assertEqual(last_checkpoint_1, last_checkpoint_other_rank_1)
        self.assertEqual(last_checkpoint_1, lc1.save_raw())
        self.assertEqual(bst_1.save_raw(), lc1.save_raw())

        # Start new training run, starting from existing model
        res_2 = {}
        bst_2 = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[
                _checkpoint_callback(frequency=1, before_iteration=True),
                _checkpoint_callback(frequency=1, before_iteration=False)
            ],
            num_boost_round=4,
            ray_params=RayParams(num_actors=2),
            additional_results=res_2,
            xgb_model=last_checkpoint_1)
        first_checkpoint_2 = res_2["callback_returns"][0][0]
        first_checkpoint_other_actor_2 = res_2["callback_returns"][1][0]
        last_checkpoint_2 = res_2["callback_returns"][0][-1]
        last_checkpoint_other_actor_2 = res_2["callback_returns"][1][-1]

        fcp_bst = xgb.Booster()
        fcp_bst.load_model(first_checkpoint_2)

        lcp_bst = xgb.Booster()
        lcp_bst.load_model(last_checkpoint_2)

        # Sanity check
        self.assertEqual(first_checkpoint_2, first_checkpoint_other_actor_2)
        self.assertEqual(last_checkpoint_2, last_checkpoint_other_actor_2)
        self.assertEqual(bst_2.save_raw(), lcp_bst.save_raw())

        # Training should not have proceeded for the first checkpoint,
        # so trees should be equal
        self.assertEqual(last_checkpoint_1, fcp_bst.save_raw())

        # Training should have proceeded for the last checkpoint,
        # so trees should not be equal
        self.assertNotEqual(fcp_bst.save_raw(), lcp_bst.save_raw())

    def testSameResultWithAndWithoutError(self):
        """Get the same model with and without errors during training."""
        # Run training
        bst_noerror = train(
            self.params,
            RayDMatrix(self.x, self.y),
            num_boost_round=10,
            ray_params=RayParams(max_actor_restarts=0, num_actors=2))

        bst_2part_1 = train(
            self.params,
            RayDMatrix(self.x, self.y),
            num_boost_round=5,
            ray_params=RayParams(max_actor_restarts=0, num_actors=2))

        bst_2part_2 = train(
            self.params,
            RayDMatrix(self.x, self.y),
            num_boost_round=5,
            ray_params=RayParams(max_actor_restarts=0, num_actors=2),
            xgb_model=bst_2part_1)

        res_error = {}
        bst_error = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[_fail_callback(self.die_lock_file, fail_iteration=7)],
            num_boost_round=10,
            ray_params=RayParams(
                max_actor_restarts=1, num_actors=2, checkpoint_frequency=5),
            additional_results=res_error)

        flat_noerror = flatten_obj({"tree": tree_obj(bst_noerror)})
        flat_error = flatten_obj({"tree": tree_obj(bst_error)})
        flat_2part = flatten_obj({"tree": tree_obj(bst_2part_2)})

        for key in flat_noerror:
            self.assertAlmostEqual(flat_noerror[key], flat_error[key])
            self.assertAlmostEqual(flat_noerror[key], flat_2part[key])

        # We fail at iteration 7, but checkpoints are saved at iteration 5
        # Thus we have two additional returns here.
        print("Callback returns:", res_error["callback_returns"][0])
        self.assertEqual(len(res_error["callback_returns"][0]), 10 + 2)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
