import unittest
from typing import Sequence
from unittest.mock import patch

import numpy as np
import pandas as pd

import ray
from ray import ObjectRef

from xgboost_ray.data_sources import Modin
from xgboost_ray.main import RayXGBoostActor

from xgboost_ray.data_sources.modin import MODIN_INSTALLED


@unittest.skipIf(
    not MODIN_INSTALLED,
    reason="Modin is not installed in a supported version.")
class ModinDataSourceTest(unittest.TestCase):
    """This test suite validates core RayDMatrix functionality."""

    def setUp(self):
        repeat = 8  # Repeat data a couple of times for stability
        self.x = np.repeat(range(8), 16).reshape((32, 4))
        self.y = np.array([0, 1, 2, 3] * repeat)

        self._init_ray()

    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

    def _init_ray(self):
        if not ray.is_initialized():
            ray.init(num_cpus=1)

    def _testAssignPartitions(self, part_nodes, actor_nodes,
                              expected_actor_parts):
        from xgboost_ray.data_sources.modin import assign_partitions_to_actors

        partitions = [
            ray.put(p) for p in np.array_split(self.x, len(part_nodes))
        ]

        # Dict from partition (obj ref) to node host
        part_to_node = dict(zip(partitions, [f"node{n}" for n in part_nodes]))
        node_to_part = [(ray.put(n), p) for p, n in part_to_node.items()]

        actors_to_node = dict(enumerate(f"node{n}" for n in actor_nodes))

        def unwrap(data, *args, **kwargs):
            return data

        with patch("modin.distributed.dataframe.pandas.unwrap_partitions"
                   ) as mocked:
            mocked.side_effect = unwrap
            actor_to_parts = assign_partitions_to_actors(
                node_to_part, actor_rank_ips=actors_to_node)

        for actor_rank, part_ids in expected_actor_parts.items():
            for i, part_id in enumerate(part_ids):
                self.assertEqual(
                    actor_to_parts[actor_rank][i],
                    partitions[part_id],
                    msg=f"Assignment failed: Actor rank {actor_rank}, "
                    f"partition {i} is not partition with ID {part_id}.")

    def _testModinAssignment(self, part_nodes, actor_nodes,
                             expected_actor_parts):
        node_ips = [
            node["NodeManagerAddress"] for node in ray.nodes() if node["Alive"]
        ]
        if len(node_ips) < max(max(actor_nodes), max(part_nodes)) + 1:
            print("Not running on cluster, skipping rest of this test.")
            return

        actor_node_ips = [node_ips[nid] for nid in actor_nodes]
        part_node_ips = [node_ips[nid] for nid in part_nodes]

        # Initialize data frames on remote nodes
        # This way we can control which partition is on which node
        @ray.remote(num_cpus=0.1)
        def create_remote_df(arr):
            return ray.put(pd.DataFrame(arr))

        partitions = np.array_split(self.x, len(part_nodes))
        node_dfs: Sequence[ObjectRef] = ray.get([
            create_remote_df.options(resources={
                f"node:{pip}": 0.1
            }).remote(partitions[pid]) for pid, pip in enumerate(part_node_ips)
        ])
        node_ip_dfs = [(ray.put(part_node_ips[pid]), node_df)
                       for pid, node_df in enumerate(node_dfs)]

        # Create modin dataframe from distributed partitions
        from modin.distributed.dataframe.pandas import (from_partitions,
                                                        unwrap_partitions)
        modin_df = from_partitions(node_ip_dfs, axis=0)

        # Sanity check
        unwrapped = unwrap_partitions(modin_df, axis=0, get_ip=True)
        ip_objs, df_objs = zip(*unwrapped)

        try:
            self.assertSequenceEqual(
                [df[0][0] for df in partitions],
                [df[0][0] for df in ray.get(list(df_objs))],
                msg="Modin mixed up the partition order")

            self.assertSequenceEqual(
                part_node_ips,
                ray.get(list(ip_objs)),
                msg="Modin moved partitions to different IPs")
        except AssertionError as exc:
            print(f"Modin part of the test failed: {exc}")
            print("This is a stochastic test failure. Ignoring the rest "
                  "of this test.")
            return

        # Create ray actors
        actors = [
            RayXGBoostActor.options(resources={
                f"node:{nip}": 0.1
            }).remote(rank=rank, num_actors=len(actor_nodes))
            for rank, nip in enumerate(actor_node_ips)
        ]

        # Calculate shards
        _, actor_to_parts = Modin.get_actor_shards(modin_df, actors)

        for actor_rank, part_ids in expected_actor_parts.items():
            for i, part_id in enumerate(part_ids):
                assigned_df = ray.get(actor_to_parts[actor_rank][i])
                part_df = pd.DataFrame(partitions[part_id])

                self.assertTrue(
                    assigned_df.equals(part_df),
                    msg=f"Assignment failed: Actor rank {actor_rank}, "
                    f"partition {i} is not partition with ID {part_id}.")

    def testAssignEvenTrivial(self):
        """Assign actors to co-located partitions, trivial case.

        In this test case, partitions are already evenly distributed across
        nodes.
        """
        part_nodes = [0, 0, 1, 1, 2, 2, 3, 3]
        actor_nodes = [0, 1, 2, 3]

        expected_actor_parts = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7],
        }
        self._testAssignPartitions(part_nodes, actor_nodes,
                                   expected_actor_parts)
        self._testModinAssignment(part_nodes, actor_nodes,
                                  expected_actor_parts)

    def testAssignEvenRedistributeOne(self):
        """Assign actors to co-located partitions, non-trivial case.

        Here two actors are located on node0, but only one can be filled
        with co-located partitions. The other one has to fetch data from
        node1.
        """
        part_nodes = [0, 0, 0, 1, 1, 1, 2, 2]
        actor_nodes = [0, 0, 1, 2]

        expected_actor_parts = {
            0: [0, 2],
            1: [1, 5],
            2: [3, 4],
            3: [6, 7],
        }
        self._testAssignPartitions(part_nodes, actor_nodes,
                                   expected_actor_parts)
        self._testModinAssignment(part_nodes, actor_nodes,
                                  expected_actor_parts)

    def testAssignEvenRedistributeMost(self):
        """Assign actors to co-located partitions, redistribute case.

        In this test case, partitions are all on one node.
        """
        part_nodes = [0, 0, 0, 0, 0, 0, 0, 0]
        actor_nodes = [0, 1, 2, 3]

        expected_actor_parts = {
            0: [0, 1],
            1: [2, 5],
            2: [3, 6],
            3: [4, 7],
        }
        self._testAssignPartitions(part_nodes, actor_nodes,
                                   expected_actor_parts)

        # This part of the test never works - Modin materializes partitions
        # onto different nodes while unwrapping.
        # self._testModinAssignment(part_nodes, actor_nodes,
        #                           expected_actor_parts)

    def testAssignUnevenTrivial(self):
        """Assign actors to co-located partitions, trivial uneven case.

        In this test case, not all actors get the same amount of partitions.
        """
        part_nodes = [0, 0, 0, 1, 1, 2, 2, 2]
        actor_nodes = [0, 1, 2]

        expected_actor_parts = {
            0: [0, 1, 2],
            1: [3, 4],
            2: [5, 6, 7],
        }
        self._testAssignPartitions(part_nodes, actor_nodes,
                                   expected_actor_parts)
        self._testModinAssignment(part_nodes, actor_nodes,
                                  expected_actor_parts)

    def testAssignUnevenRedistribute(self):
        """Assign actors to co-located partitions, redistribute uneven case.

        In this test case, not all actors get the same amount of partitions.
        Some actors have to fetch partitions from other nodes
        """
        part_nodes = [0, 0, 1, 1, 1, 1, 2, 3]
        actor_nodes = [0, 1, 2]

        expected_actor_parts = {
            0: [0, 1, 5],
            1: [2, 3, 4],
            2: [6, 7],
        }
        self._testAssignPartitions(part_nodes, actor_nodes,
                                   expected_actor_parts)
        self._testModinAssignment(part_nodes, actor_nodes,
                                  expected_actor_parts)

    def testAssignUnevenRedistributeColocated(self):
        """Assign actors to co-located partitions, redistribute uneven case.

        Here we have an uneven split of partitions. One actor does not get
        a co-located shard assigned in favor for a non-coloated actor.
        """
        part_nodes = [0, 0, 0, 0, 0, 0, 0]
        actor_nodes = [0, 0, 1]

        expected_actor_parts = {
            0: [0, 2, 4],
            1: [1, 3],
            2: [5, 6],
        }
        self._testAssignPartitions(part_nodes, actor_nodes,
                                   expected_actor_parts)
        self._testModinAssignment(part_nodes, actor_nodes,
                                  expected_actor_parts)

    def testAssignUnevenRedistributeAll(self):
        """Assign actors to co-located partitions, redistribute uneven case.

        In this test case, not all actors get the same amount of partitions.
        Some actors have to fetch partitions from other nodes
        """
        part_nodes = [1, 1, 1, 1, 0, 0, 0]
        actor_nodes = [1, 1, 2]

        expected_actor_parts = {
            0: [0, 2, 4],
            1: [1, 3],
            2: [5, 6],
        }
        self._testAssignPartitions(part_nodes, actor_nodes,
                                   expected_actor_parts)
        self._testModinAssignment(part_nodes, actor_nodes,
                                  expected_actor_parts)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
