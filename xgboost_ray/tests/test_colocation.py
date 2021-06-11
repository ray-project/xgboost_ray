import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import pytest

import numpy as np

import ray
from xgboost_ray import train, RayDMatrix, RayParams
from xgboost_ray.main import _train
from xgboost_ray.util import _EventActor, _QueueActor


class _MockQueueActor(_QueueActor):
    def get_node_id(self):
        return ray.state.current_node_id()


class _MockEventActor(_EventActor):
    def get_node_id(self):
        return ray.state.current_node_id()


@pytest.mark.usefixtures("ray_start_cluster")
class TestColocation(unittest.TestCase):
    def setUp(self) -> None:
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

    def tearDown(self) -> None:
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        ray.shutdown()

    @patch("xgboost_ray.util._QueueActor", _MockQueueActor)
    @patch("xgboost_ray.util._EventActor", _MockEventActor)
    def test_communication_colocation(self):
        """Checks that Queue and Event actors are colocated with the driver."""
        with self.ray_start_cluster() as cluster:
            cluster.add_node(num_cpus=3)
            cluster.add_node(num_cpus=3)
            cluster.wait_for_nodes()
            ray.init(address=cluster.address)

            local_node = ray.state.current_node_id()

            # Note that these will have the same IP in the test cluster
            assert len(ray.state.node_ids()) == 2
            assert local_node in ray.state.node_ids()

            def _mock_train(*args, _training_state, **kwargs):
                assert ray.get(_training_state.queue.actor.get_node_id.remote(
                )) == ray.state.current_node_id()
                assert ray.get(
                    _training_state.stop_event.actor.get_node_id.remote()) == \
                    ray.state.current_node_id()
                return _train(*args, _training_state=_training_state, **kwargs)

            with patch("xgboost_ray.main._train") as mocked:
                mocked.side_effect = _mock_train
                train(
                    self.params,
                    RayDMatrix(self.x, self.y),
                    num_boost_round=2,
                    ray_params=RayParams(max_actor_restarts=1, num_actors=6))

    def test_no_tune_spread(self):
        """Tests whether workers are spread when not using Tune."""
        with self.ray_start_cluster() as cluster:
            cluster.add_node(num_cpus=2)
            cluster.add_node(num_cpus=2)
            cluster.wait_for_nodes()
            ray.init(address=cluster.address)

            ray_params = RayParams(
                max_actor_restarts=1, num_actors=2, cpus_per_actor=2)

            def _mock_train(*args, _training_state, **kwargs):
                try:
                    results = _train(
                        *args, _training_state=_training_state, **kwargs)
                    return results
                except Exception:
                    raise
                finally:
                    assert len(_training_state.actors) == 2
                    if not any(a is None for a in _training_state.actors):
                        actor_infos = ray.state.actors()
                        actor_nodes = []
                        for a in _training_state.actors:
                            actor_info = actor_infos.get(a._actor_id.hex())
                            actor_node = actor_info["Address"]["NodeID"]
                            actor_nodes.append(actor_node)
                        assert actor_nodes[0] != actor_nodes[1]

            with patch("xgboost_ray.main._train", _mock_train):
                train(
                    self.params,
                    RayDMatrix(self.x, self.y),
                    num_boost_round=4,
                    ray_params=ray_params)

    def test_tune_pack(self):
        """Tests whether workers are packed when using Tune."""
        try:
            from ray import tune
        except ImportError:
            self.skipTest("Tune is not installed.")
            return
        with self.ray_start_cluster() as cluster:
            num_actors = 2
            cluster.add_node(num_cpus=3)
            cluster.add_node(num_cpus=3)
            ray.init(address=cluster.address)

            ray_params = RayParams(
                max_actor_restarts=1, num_actors=num_actors, cpus_per_actor=1)

            def _mock_train(*args, _training_state, **kwargs):
                try:
                    results = _train(
                        *args, _training_state=_training_state, **kwargs)
                    return results
                except Exception:
                    raise
                finally:
                    assert len(_training_state.actors) == num_actors
                    if not any(a is None for a in _training_state.actors):
                        actor_infos = ray.state.actors()
                        actor_nodes = []
                        for a in _training_state.actors:
                            actor_info = actor_infos.get(a._actor_id.hex())
                            actor_node = actor_info["Address"]["NodeID"]
                            actor_nodes.append(actor_node)
                        assert actor_nodes[0] == actor_nodes[1]

            def train_func(params, x, y, ray_params):
                def inner_func(config):
                    with patch("xgboost_ray.main._train", _mock_train):
                        train(
                            params,
                            RayDMatrix(x, y),
                            num_boost_round=4,
                            ray_params=ray_params)

                return inner_func

            tune.run(
                train_func(self.params, self.x, self.y, ray_params),
                resources_per_trial=ray_params.get_tune_resources(),
                num_samples=1,
            )


if __name__ == "__main__":
    import pytest  # noqa: F811
    import sys
    sys.exit(pytest.main(["-v", __file__]))
