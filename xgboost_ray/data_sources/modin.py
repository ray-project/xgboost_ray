from typing import Any, Optional, Sequence, Dict, Union, Tuple

from collections import defaultdict
import itertools
import math
import pandas as pd

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.object_store import ObjectStore

try:
    import modin  # noqa: F401
    MODIN_INSTALLED = True
except ImportError:
    MODIN_INSTALLED = False


def _assert_modin_installed():
    if not MODIN_INSTALLED:
        raise RuntimeError(
            "Tried to use Modin as a data source, but modin is not "
            "installed. This function shouldn't have been called. "
            "\nFIX THIS by installing modin: `pip install modin`. "
            "\nPlease also raise an issue on our GitHub: "
            "https://github.com/ray-project/xgboost_ray as this part of "
            "the code should not have been reached.")


class Modin(DataSource):
    """Read from distributed Modin dataframe.

    `Modin <https://github.com/modin-project/modin>`_ is a distributed
    drop-in replacement for pandas supporting Ray as a backend.

    Modin dataframes are stored on multiple actors, making them
    suitable for distributed loading.
    """
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        if not MODIN_INSTALLED:
            return False
        from modin.pandas import DataFrame as ModinDataFrame, \
            Series as ModinSeries

        return isinstance(data, (ModinDataFrame, ModinSeries))

    @staticmethod
    def load_data(
            data: Any,  # modin.pandas.DataFrame
            ignore: Optional[Sequence[str]] = None,
            indices: Optional[Union[Sequence[int], Sequence[
                ObjectRef]]] = None,
            **kwargs) -> pd.DataFrame:
        _assert_modin_installed()

        if indices is not None and len(indices) > 0 and isinstance(
                indices[0], ObjectRef):
            # We got a list of ObjectRefs belonging to Modin partitions
            return ObjectStore.load_data(
                data=indices, indices=None, ignore=ignore)

        local_df = data
        if indices:
            local_df = local_df.iloc[indices]

        local_df = local_df._to_pandas()

        if ignore:
            local_df = local_df[local_df.columns.difference(ignore)]

        return local_df

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        _assert_modin_installed()
        from modin.pandas import DataFrame as ModinDataFrame, \
            Series as ModinSeries

        if isinstance(data, ModinDataFrame):
            return pd.Series(data._to_pandas().squeeze())
        elif isinstance(data, ModinSeries):
            return data._to_pandas()

        return DataSource.convert_to_series(data)

    @staticmethod
    def get_actor_shards(
            data: Any,  # modin.pandas.DataFrame
            actors: Sequence[ActorHandle]) -> \
            Tuple[Any, Optional[Dict[int, Any]]]:
        _assert_modin_installed()
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
        # Modin dataframes are not serializable, so pass None here
        # as the first return value
        return None, assign_partitions_to_actors(data, actor_rank_ips)

    @staticmethod
    def get_n(data: Any):
        """
        For naive distributed loading we just return the number of rows
        here. Loading by shard is achieved via `get_actor_shards()`
        """
        return len(data)


def assign_partitions_to_actors(data: Any, actor_rank_ips: Dict[int, str]) \
        -> Dict[int, Sequence[ObjectRef]]:
    """Assign partitions from a Modin dataframe to actors.

    This function collects the Modin partitions and evenly distributes
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
    from modin.distributed.dataframe.pandas import unwrap_partitions

    unwrapped = unwrap_partitions(data, axis=0, get_ip=True)

    ip_objs, part_objs = zip(*unwrapped)

    # Build a table mapping from IP to list of partitions
    ip_to_parts = defaultdict(list)
    for ip, part_obj in zip(ray.get(list(ip_objs)), part_objs):
        ip_to_parts[ip].append(part_obj)

    num_partitions = len(part_objs)
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
            f"There are still partitions left to assign, but no actor "
            f"has capacity for more. This is probably a bug. Please go "
            f"to https://github.com/ray-project/xgboost_ray to report it.")

    return actor_to_partitions
