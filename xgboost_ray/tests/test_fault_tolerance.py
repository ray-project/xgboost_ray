import os
import random
import shutil
import tempfile
import time

import numpy as np
import unittest
import xgboost as xgb

import ray

from xgboost_ray import train, RayDMatrix


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
        self.y = np.array([
            0, 1, 2, 3
        ] * repeat)

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

        ray.init(num_cpus=2, num_gpus=0)

    def tearDown(self) -> None:
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        ray.shutdown()

    @staticmethod
    def _fail_callback(die_lock_file: str):
        def callback(env):
            if env.iteration == 6 and not os.path.exists(die_lock_file):
                pass
                # By sleeping a random amount of time we make sure only
                # one worker dies.
                time.sleep(random.uniform(0.5, 2.0))
                if os.path.exists(die_lock_file):
                    return

                with open(die_lock_file, "wt") as fp:
                    fp.write("")
                import sys
                sys.exit(1)
        return callback

    def testTrainingContinuation(self):
        """This should continue after one actor died."""
        bst, _ = train(
            self.params,
            RayDMatrix(self.x, self.y),
            callbacks=[self._fail_callback(self.die_lock_file)],
            num_boost_round=20,
            max_actor_restarts=1,
            num_actors=2,
            checkpoint_path=self.tmpdir)

        x_mat = xgb.DMatrix(self.x, feature_names=["0", "1", "2", "3"])
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))

    def testTrainingStop(self):
        """This should now continue training after one actor died."""
        # The `train()` function raises a RuntimeError
        with self.assertRaises(RuntimeError):
            bst, _ = train(
                self.params,
                RayDMatrix(self.x, self.y),
                callbacks=[self._fail_callback(self.die_lock_file)],
                num_boost_round=20,
                max_actor_restarts=0,
                num_actors=2,
                checkpoint_path=self.tmpdir)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
