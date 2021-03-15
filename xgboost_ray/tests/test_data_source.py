import unittest
from unittest.mock import patch

import numpy as np

import ray


class ModinDataSourceTest(unittest.TestCase):
    """This test suite validates core RayDMatrix functionality."""

    def setUp(self):
        repeat = 8  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        self.y = np.array([0, 1, 2, 3] * repeat)

    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=1, local_mode=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def _testAssignPartitions(self, part_nodes, actors_to_node,
                              expected_actor_parts):
        from xgboost_ray.data_sources.modin import assign_partitions_to_actors

        partitions = [ray.put(p) for p in np.split(self.x, len(part_nodes))]

        part_to_node = dict(zip(partitions, part_nodes))
        node_to_part = [(ray.put(n), p) for p, n in part_to_node.items()]

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

    def testAssignEvenTrivial(self):
        """Assign actors to co-located partitions, trivial case.

        In this test case, partitions are already evenly distributed across
        nodes.
        """
        part_nodes = [
            "node0",
            "node0",
            "node1",
            "node1",
            "node2",
            "node2",
            "node3",
            "node3",
        ]
        actors_to_node = {
            0: "node0",
            1: "node1",
            2: "node2",
            3: "node3",
        }
        expected_actor_parts = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7],
        }
        self._testAssignPartitions(part_nodes, actors_to_node,
                                   expected_actor_parts)

    def testAssignEvenRedistributeOne(self):
        """Assign actors to co-located partitions, non-trivial case.

        Here two actors are located on node0, but only one can be filled
        with co-located partitions. The other one has to fetch data from
        node1.
        """
        part_nodes = [
            "node0",
            "node0",
            "node0",
            "node1",
            "node1",
            "node1",
            "node2",
            "node2",
        ]
        actors_to_node = {
            0: "node0",
            1: "node0",
            2: "node1",
            3: "node2",
        }
        expected_actor_parts = {
            0: [0, 2],
            1: [1, 5],
            2: [3, 4],
            3: [6, 7],
        }
        self._testAssignPartitions(part_nodes, actors_to_node,
                                   expected_actor_parts)

    def testAssignEvenRedistributeMost(self):
        """Assign actors to co-located partitions, redistribute case.

        In this test case, partitions are all on one node.
        """
        part_nodes = [
            "node0",
            "node0",
            "node0",
            "node0",
            "node0",
            "node0",
            "node0",
            "node0",
        ]
        actors_to_node = {
            0: "node0",
            1: "node1",
            2: "node2",
            3: "node3",
        }
        expected_actor_parts = {
            0: [0, 1],
            1: [2, 5],
            2: [3, 6],
            3: [4, 7],
        }
        self._testAssignPartitions(part_nodes, actors_to_node,
                                   expected_actor_parts)

    def testAssignUnevenTrivial(self):
        """Assign actors to co-located partitions, trivial uneven case.

        In this test case, not all actors get the same amount of partitions.
        """
        part_nodes = [
            "node0",
            "node0",
            "node0",
            "node1",
            "node1",
            "node2",
            "node2",
            "node2",
        ]
        actors_to_node = {
            0: "node0",
            1: "node1",
            2: "node2",
        }
        expected_actor_parts = {
            0: [0, 1, 2],
            1: [3, 4],
            2: [5, 6, 7],
        }
        self._testAssignPartitions(part_nodes, actors_to_node,
                                   expected_actor_parts)

    def testAssignUnevenRedistribute(self):
        """Assign actors to co-located partitions, redistribute uneven case.

        In this test case, not all actors get the same amount of partitions.
        Some actors have to fetch partitions from other nodes
        """
        part_nodes = [
            "node0",
            "node0",
            "node1",
            "node1",
            "node1",
            "node1",
            "node2",
            "node4",
        ]
        actors_to_node = {
            0: "node0",
            1: "node1",
            2: "node2",
        }
        expected_actor_parts = {
            0: [0, 1, 5],
            1: [2, 3, 4],
            2: [6, 7],
        }
        self._testAssignPartitions(part_nodes, actors_to_node,
                                   expected_actor_parts)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
