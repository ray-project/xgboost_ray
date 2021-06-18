import logging
import os
import shutil
import tempfile
import time
from unittest.mock import patch, DEFAULT, MagicMock

import numpy as np
import unittest
import xgboost as xgb

import ray

from xgboost_ray import train, RayDMatrix, RayParams
from xgboost_ray.main import RayXGBoostActorAvailable
from xgboost_ray.tests.fault_tolerance import FaultToleranceManager, \
    DelayedLoadingCallback, DieCallback
from xgboost_ray.tests.utils import flatten_obj, _checkpoint_callback, \
    _fail_callback, tree_obj, _kill_callback, get_num_trees


class _FakeTask(MagicMock):
    ready = False

    def is_ready(self):
        return self.ready


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
            "num_class": 4
        }

        self.tmpdir = str(tempfile.mkdtemp())

        self.die_lock_file = "/tmp/died_worker.lock"
        if os.path.exists(self.die_lock_file):
            os.remove(self.die_lock_file)

        self.die_lock_file_2 = "/tmp/died_worker_2.lock"
        if os.path.exists(self.die_lock_file_2):
            os.remove(self.die_lock_file_2)

        ray.init(num_cpus=2, num_gpus=0, log_to_driver=True)

    def tearDown(self) -> None:
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        ray.shutdown()

        if os.path.exists(self.die_lock_file):
            os.remove(self.die_lock_file)

        if os.path.exists(self.die_lock_file_2):
            os.remove(self.die_lock_file_2)

    def testTrainingContinuationKilled(self):
        """This should continue after one actor died."""
        additional_results = {}
        keep_actors = {}

        def keep(actors, *args, **kwargs):
            keep_actors["actors"] = actors.copy()
            return DEFAULT

        with patch("xgboost_ray.main._shutdown") as mocked:
            mocked.side_effect = keep
            bst = train(
                self.params,
                RayDMatrix(self.x, self.y),
                callbacks=[_kill_callback(self.die_lock_file)],
                num_boost_round=20,
                ray_params=RayParams(max_actor_restarts=1, num_actors=2),
                additional_results=additional_results)

        self.assertEqual(20, get_num_trees(bst))

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

        actors = keep_actors["actors"]
        # End with two working actors
        self.assertTrue(actors[0])
        self.assertTrue(actors[1])

        # Two workers finished, so N=32
        self.assertEqual(additional_results["total_n"], 32)

    @patch("xgboost_ray.main.ELASTIC_RESTART_DISABLED", True)
    def testTrainingContinuationElasticKilled(self):
        """This should continue after one actor died."""
        logging.getLogger().setLevel(10)

        additional_results = {}
        keep_actors = {}

        def keep(actors, *args, **kwargs):
            keep_actors["actors"] = actors.copy()
            return DEFAULT

        with patch("xgboost_ray.main._shutdown") as mocked:
            mocked.side_effect = keep
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
                additional_results=additional_results)

        self.assertEqual(20, get_num_trees(bst))

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

        actors = keep_actors["actors"]
        # First actor does not get recreated
        self.assertEqual(actors[0], None)
        self.assertTrue(actors[1])

        # Only one worker finished, so n=16
        self.assertEqual(additional_results["total_n"], 16)

    @patch("xgboost_ray.main.ELASTIC_RESTART_DISABLED", False)
    def testTrainingContinuationElasticKilledRestarted(self):
        """This should continue after one actor died and restart it."""
        logging.getLogger().setLevel(10)

        ft_manager = FaultToleranceManager.remote()

        ft_manager.schedule_kill.remote(rank=0, boost_round=6)
        ft_manager.delay_return.remote(
            rank=1, start_boost_round=12, end_boost_round=21)

        delay_callback = DelayedLoadingCallback(
            ft_manager, reload_data=True, sleep_time=0.1)
        die_callback = DieCallback(ft_manager, training_delay=0.25)

        additional_results = {}
        keep_actors = {}

        def keep(actors, *args, **kwargs):
            keep_actors["actors"] = actors.copy()
            return DEFAULT

        with patch("xgboost_ray.main._shutdown") as mocked:
            mocked.side_effect = keep
            bst = train(
                self.params,
                RayDMatrix(self.x, self.y),
                callbacks=[die_callback],
                num_boost_round=20,
                ray_params=RayParams(
                    max_actor_restarts=1,
                    num_actors=2,
                    elastic_training=True,
                    max_failed_actors=1,
                    distributed_callbacks=[delay_callback]),
                additional_results=additional_results)

        self.assertEqual(20, get_num_trees(bst))

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

        actors = keep_actors["actors"]

        # First actor gets recreated
        self.assertTrue(actors[0])
        self.assertTrue(actors[1])

        # Both workers finished, so n=32
        self.assertEqual(additional_results["total_n"], 32)

    @patch("xgboost_ray.main.ELASTIC_RESTART_DISABLED", True)
    def testTrainingContinuationElasticMultiKilled(self):
        """This should still show 20 boost rounds after two failures."""
        logging.getLogger().setLevel(10)

        additional_results = {}

        bst = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[
                _kill_callback(
                    self.die_lock_file, fail_iteration=6, actor_rank=0),
                _kill_callback(
                    self.die_lock_file_2, fail_iteration=14, actor_rank=1),
            ],
            num_boost_round=20,
            ray_params=RayParams(
                max_actor_restarts=2,
                num_actors=2,
                elastic_training=True,
                max_failed_actors=2),
            additional_results=additional_results)

        self.assertEqual(20, get_num_trees(bst))

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

    @patch("xgboost_ray.main.ELASTIC_RESTART_DISABLED", True)
    def testTrainingContinuationElasticFailed(self):
        """This should continue after one actor failed training."""

        additional_results = {}
        keep_actors = {}

        def keep(actors, *args, **kwargs):
            keep_actors["actors"] = actors.copy()
            return DEFAULT

        with patch("xgboost_ray.main._shutdown") as mocked:
            mocked.side_effect = keep
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
                additional_results=additional_results)

        self.assertEqual(20, get_num_trees(bst))

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))
        print(f"Got correct predictions: {pred_y}")

        actors = keep_actors["actors"]
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
                callbacks=[_kill_callback(self.die_lock_file)],
                num_boost_round=20,
                ray_params=RayParams(max_actor_restarts=0, num_actors=2))

    def testTrainingStopElastic(self):
        """This should now stop training after one actor died."""
        # The `train()` function raises a RuntimeError
        ft_manager = FaultToleranceManager.remote()

        ft_manager.schedule_kill.remote(rank=0, boost_round=3)
        ft_manager.schedule_kill.remote(rank=1, boost_round=6)
        ft_manager.delay_return.remote(
            rank=0, start_boost_round=4, end_boost_round=5)

        delay_callback = DelayedLoadingCallback(
            ft_manager, reload_data=True, sleep_time=0.1)
        die_callback = DieCallback(ft_manager, training_delay=0.25)

        with self.assertRaises(RuntimeError):
            train(
                self.params,
                RayDMatrix(self.x, self.y),
                callbacks=[die_callback],
                num_boost_round=20,
                ray_params=RayParams(
                    elastic_training=True,
                    max_failed_actors=1,
                    max_actor_restarts=1,
                    num_actors=2,
                    distributed_callbacks=[delay_callback]))

    def testCheckpointContinuationValidity(self):
        """Test that checkpoints are stored and loaded correctly"""

        # Train once, get checkpoint via callback returns
        res_1 = {}
        bst_1 = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[
                _checkpoint_callback(frequency=1, before_iteration_=False)
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
                _checkpoint_callback(frequency=1, before_iteration_=True),
                _checkpoint_callback(frequency=1, before_iteration_=False)
            ],
            num_boost_round=4,
            ray_params=RayParams(num_actors=2),
            additional_results=res_2,
            xgb_model=lc1)
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

    @patch("xgboost_ray.main._PrepareActorTask", _FakeTask)
    @patch("xgboost_ray.elastic._PrepareActorTask", _FakeTask)
    @patch("xgboost_ray.main._RemoteRayXGBoostActor", MagicMock)
    @patch("xgboost_ray.main.ELASTIC_RESTART_GRACE_PERIOD_S", 30)
    @patch("xgboost_ray.elastic.ELASTIC_RESTART_GRACE_PERIOD_S", 30)
    def testMaybeScheduleNewActors(self):
        """Test scheduling of new actors if resources become available.

        Context: We are training with num_actors=8, of which 3 actors are
        dead. The cluster has resources to restart 2 of these actors.

        In this test, we walk through the `_maybe_schedule_new_actors` and
        `_update_scheduled_actor_states` methods, checking their state
        after each call.

        """
        from xgboost_ray.main import _TrainingState
        from xgboost_ray.elastic import _update_scheduled_actor_states
        from xgboost_ray.elastic import _maybe_schedule_new_actors

        # Three actors are dead
        actors = [
            MagicMock(), None,
            MagicMock(),
            MagicMock(), None,
            MagicMock(), None,
            MagicMock()
        ]

        # Mock training state
        state = _TrainingState(
            actors=actors,
            queue=MagicMock(),
            stop_event=MagicMock(),
            checkpoint=MagicMock(),
            additional_results={},
            failed_actor_ranks=set(),
        )

        created_actors = []

        def fake_create_actor(rank, *args, **kwargs):
            created_actors.append(rank)
            return MagicMock()

        with patch("xgboost_ray.elastic._create_actor") as create_actor:
            create_actor.side_effect = fake_create_actor

            _maybe_schedule_new_actors(
                training_state=state,
                num_cpus_per_actor=8,
                num_gpus_per_actor=0,
                resources_per_actor={"custom": 1.0},
                load_data=[],
                ray_params=RayParams(
                    num_actors=8,
                    elastic_training=True,
                    max_failed_actors=1,
                    max_actor_restarts=2))

            # 3 new actors should have been created
            self.assertEqual(len(created_actors), 3)
            self.assertEqual(len(state.pending_actors), 3)

            # The number of created actors shouldn't change even
            # if we run this function again.
            _maybe_schedule_new_actors(
                training_state=state,
                num_cpus_per_actor=8,
                num_gpus_per_actor=0,
                resources_per_actor={"custom": 1.0},
                load_data=[],
                ray_params=RayParams(
                    num_actors=8,
                    elastic_training=True,
                    max_failed_actors=1,
                    max_actor_restarts=2))

            self.assertEqual(len(created_actors), 3)
            self.assertEqual(len(state.pending_actors), 3)

            # The actors have not yet been promoted because the
            # loading task has not finished.
            self.assertFalse(actors[1])
            self.assertFalse(actors[4])
            self.assertFalse(actors[6])

            # Update status, nothing should change
            _update_scheduled_actor_states(training_state=state)

            self.assertFalse(actors[1])
            self.assertFalse(actors[4])
            self.assertFalse(actors[6])

            # Set loading task status to finished, but only for first actor
            for _, (_, task) in state.pending_actors.items():
                task.ready = True
                break

            # Update status. This shouldn't raise RayXGBoostActorAvailable
            # because we still have a grace period to wait for the second
            # actor.
            _update_scheduled_actor_states(training_state=state)

            # Grace period is set through ELASTIC_RESTART_GRACE_PERIOD_S
            # Allow for some slack in test execution
            self.assertGreaterEqual(state.restart_training_at,
                                    time.time() + 22)

            # The first actor should have been promoted to full actor
            self.assertTrue(actors[1])
            self.assertFalse(actors[4])
            self.assertFalse(actors[6])

            # Set loading task status to finished for all actors
            for _, (_, task) in state.pending_actors.items():
                task.ready = True

            # Update status. This should now raise RayXGBoostActorAvailable
            # immediately as there are no pending actors left to wait for.
            with self.assertRaises(RayXGBoostActorAvailable):
                _update_scheduled_actor_states(training_state=state)

            # All restarted actors should have been promoted to full actors
            self.assertTrue(actors[1])
            self.assertTrue(actors[4])
            self.assertTrue(actors[6])

    def testFaultToleranceManager(self):
        ft_manager = FaultToleranceManager.remote()

        ft_manager.schedule_kill.remote(rank=1, boost_round=16)
        ft_manager.delay_return.remote(
            rank=1, start_boost_round=14, end_boost_round=68)

        delay_callback = DelayedLoadingCallback(
            ft_manager, reload_data=True, sleep_time=0.1)
        die_callback = DieCallback(ft_manager, training_delay=0.25)

        res_1 = {}
        train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[die_callback],
            num_boost_round=100,
            ray_params=RayParams(
                num_actors=2,
                checkpoint_frequency=1,
                elastic_training=True,
                max_failed_actors=1,
                max_actor_restarts=1,
                distributed_callbacks=[delay_callback]),
            additional_results=res_1)

        logs = ray.get(ft_manager.get_logs.remote())

        print(logs)

        self.assertSequenceEqual([g for g, _ in logs[0][0:99]], range(99))

        # Which steps exactly are executed is stochastic. The rank 1 actor
        # will die at iteration 16, so at least 15 will be logged (though 16
        # might be logged as well). Iterations 17 to 67 should never be
        # logged. It comes back some time after iteration 68, so this might
        # be iter 68, 69, or later. We just make sure it comes back at all
        # (iter 70+).
        global_steps = [g for g, _ in logs[1]]
        self.assertTrue(global_steps)
        self.assertIn(15, global_steps)
        self.assertNotIn(17, global_steps)
        self.assertNotIn(67, global_steps)
        self.assertIn(75, global_steps)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
