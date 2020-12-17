import os
import shutil
import tempfile
from unittest.mock import patch
import numpy as np

import ray
from xgboost_ray import train, RayDMatrix, RayParams
from xgboost_ray.main import _train
from xgboost_ray.tests.test_fault_tolerance import _kill_callback
from xgboost_ray.util import _EventActor, _QueueActor


class _MockQueueActor(_QueueActor):
    def get_node_id(self):
        return ray.state.current_node_id()

class _MockEventActor(_EventActor):
    def get_node_id(self):
        return ray.state.current_node_id()


class TestColocation:
    def setup_method(self):
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

        self.tmpdir = str(tempfile.mkdtemp())

        self.die_lock_file = "/tmp/died_worker.lock"
        if os.path.exists(self.die_lock_file):
            os.remove(self.die_lock_file)

    def teardown_method(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        ray.shutdown()

    @patch("xgboost_ray.util._QueueActor", _MockQueueActor)
    @patch("xgboost_ray.util._EventActor", _MockEventActor)
    def test_communication_colocation(self, ray_start_cluster):
        # Make sure that the Queue and Event actors are colocated with the driver.
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=3)
        cluster.add_node(num_cpus=3)
        ray.init(address=cluster.address)

        local_node = ray.state.current_node_id()

        # Note that these will have the same IP in the test cluster
        assert len(ray.state.node_ids()) == 2
        assert local_node in ray.state.node_ids()


        def _mock_train(*args, _queue, _stop_event, **kwargs):
            assert ray.get(
                _queue.actor.get_node_id.remote()) == ray.state.current_node_id()
            assert ray.get(
                _stop_event.actor.get_node_id.remote()) == \
                   ray.state.current_node_id()
            return _train(*args, _queue=_queue, _stop_event=_stop_event, **kwargs)

        with patch("xgboost_ray.main._train", _mock_train):
            bst = train(
                self.params,
                RayDMatrix(self.x, self.y),
                callbacks=[_kill_callback(self.die_lock_file,
                                          fail_iteration=1)],
                num_boost_round=2,
                ray_params=RayParams(max_actor_restarts=1, num_actors=6))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))