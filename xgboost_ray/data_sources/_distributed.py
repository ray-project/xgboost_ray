import itertools
import math
from collections import defaultdict
from typing import Dict, Any, Sequence

import ray
from ray.actor import ActorHandle


def get_actor_rank_ips(actors: Sequence[ActorHandle]) -> Dict[int, str]:
    """Get a dict mapping from actor ranks to their IPs"""
    no_obj = ray.put(None)
    # Build a dict mapping actor ranks to their IP addresses
    actor_rank_ips: Dict[int, str] = {
        rank: ip
        for rank, ip in enumerate(
            ray.get([
                actor.ip.remote() if actor is not None else no_obj
                for actor in actors
            ]))
    }
    return actor_rank_ips


def assign_partitions_to_actors(
        ip_to_parts: Dict[int, Any],
        actor_rank_ips: Dict[int, str]) -> Dict[int, Sequence[Any]]:
    """Assign partitions from a distributed dataframe to actors.

    This function collects distributed partitions and evenly distributes
    them to actors, trying to minimize data transfer by respecting
    co-locality.

    This function currently does _not_ take partition sizes into account
    for distributing data. It assumes that all partitions have (more or less)
    the same length.

    Instead, partitions are evenly distributed. E.g. for 8 partitions and 3
    actors, each actor gets assigned 2 or 3 partitions. Which partitions are
    assigned depends on the data locality.

    The algorithm is as follows: For any number of data partitions, get the
    Ray object references to the shards and the IP addresses where they
    currently live.

    Calculate the minimum and maximum amount of partitions per actor. These
    numbers should differ by at most 1. Also calculate how many actors will
    get more partitions assigned than the other actors.

    First, each actor gets assigned up to ``max_parts_per_actor`` co-located
    partitions. Only up to ``num_actors_with_max_parts`` actors get the
    maximum number of partitions, the rest try to fill the minimum.

    The rest of the partitions (all of which cannot be assigned to a
    co-located actor) are assigned to actors until there are none left.
    """
    num_partitions = sum(len(parts) for parts in ip_to_parts.values())
    num_actors = len(actor_rank_ips)
    min_parts_per_actor = max(0, math.floor(num_partitions / num_actors))
    max_parts_per_actor = max(1, math.ceil(num_partitions / num_actors))
    num_actors_with_max_parts = num_partitions % num_actors

    # This is our result dict that maps actor objects to a list of partitions
    actor_to_partitions = defaultdict(list)

    # First we loop through the actors and assign them partitions from their
    # own IPs. Do this until each actor has `min_parts_per_actor` partitions
    partition_assigned = True
    while partition_assigned:
        partition_assigned = False

        # Loop through each actor once, assigning
        for rank, actor_ip in actor_rank_ips.items():
            num_parts_left_on_ip = len(ip_to_parts[actor_ip])
            num_actor_parts = len(actor_to_partitions[rank])

            if num_parts_left_on_ip > 0 and \
               num_actor_parts < max_parts_per_actor:
                if num_actor_parts >= min_parts_per_actor:
                    # Only allow up to `num_actors_with_max_parts actors to
                    # have the maximum number of partitions assigned.
                    if num_actors_with_max_parts <= 0:
                        continue
                    num_actors_with_max_parts -= 1
                actor_to_partitions[rank].append(ip_to_parts[actor_ip].pop(0))
                partition_assigned = True

    # The rest of the partitions, no matter where they are located, could not
    # be assigned to co-located actors. Thus, we assign them
    # to actors who still need partitions.
    rest_parts = list(itertools.chain(*ip_to_parts.values()))
    partition_assigned = True
    while len(rest_parts) > 0 and partition_assigned:
        partition_assigned = False
        for rank in actor_rank_ips:
            num_actor_parts = len(actor_to_partitions[rank])
            if num_actor_parts < max_parts_per_actor:
                if num_actor_parts >= min_parts_per_actor:
                    if num_actors_with_max_parts <= 0:
                        continue
                    num_actors_with_max_parts -= 1
                actor_to_partitions[rank].append(rest_parts.pop(0))
                partition_assigned = True
            if len(rest_parts) <= 0:
                break

    if len(rest_parts) != 0:
        raise RuntimeError(
            "There are still partitions left to assign, but no actor "
            "has capacity for more. This is probably a bug. Please go "
            "to https://github.com/ray-project/xgboost_ray to report it.")

    return actor_to_partitions
