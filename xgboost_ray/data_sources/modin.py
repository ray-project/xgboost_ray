from typing import Any, Optional, Sequence

from collections import defaultdict
import itertools
import math
import pandas as pd

import ray

from xgboost_ray.data_sources.data_source import DataSource, RayFileType

try:
    import modin  # noqa: F401
    MODIN_INSTALLED = True
except ImportError:
    MODIN_INSTALLED = False


class Modin(DataSource):
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
            indices: Optional[Sequence[int]] = None,
            **kwargs) -> pd.DataFrame:
        local_df = data
        if indices:
            local_df = local_df.iloc(indices)

        local_df = local_df._to_pandas()

        if ignore:
            local_df = local_df[local_df.columns.difference(ignore)]

        return local_df

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        from modin.pandas import DataFrame as ModinDataFrame, \
            Series as ModinSeries

        if isinstance(data, ModinDataFrame):
            return pd.Series(data._to_pandas().squeeze())
        elif isinstance(data, ModinSeries):
            return data._to_pandas()

        return DataSource.convert_to_series(data)


def assign_partitions_to_actors(data, actors):
    from modin.distributed.dataframe.pandas import unwrap_partitions

    unwrapped = unwrap_partitions(data, axis=0, get_ip=True)

    ip_objs, part_objs = zip(*unwrapped)

    # Build a table mapping from IP to list of partitions
    ip_to_parts = defaultdict(list)
    for ip, part_obj in zip(ray.get(ip_objs), part_objs):
        ip_to_parts[ip].append(part_obj)

    num_partitions = len(part_objs)
    num_actors = len(actors)
    min_parts_per_actor = max(0, math.floor(num_partitions / num_actors))
    max_parts_per_actor = max(1, math.ceil(num_partitions / num_actors))

    actor_ips = dict()  # Todo

    # This is our result dict that maps actor objects to a list of partitions
    actor_to_partitions = defaultdict(list)

    # First we loop through the actors and assign them partitions from their
    # own IPs
    partition_assigned = True
    while partition_assigned:
        partition_assigned = False

        for actor in actors:
            actor_ip = actor_ips[actor]
            num_parts_left_on_ip = len(ip_to_parts[actor_ip])
            num_actor_parts = len(actor_to_partitions[actor])

            if num_parts_left_on_ip > 0 and \
                min_parts_per_actor <= num_actor_parts < max_parts_per_actor:
                actor_to_partitions[actor].append(ip_to_parts[actor_ip].pop(0))
                partition_assigned = True

    # The rest of the partitions, no matter where they are located, could not
    # be assigned to co-located actors. Thus, we assign them
    # to actors who still need partitions.
    rest_parts = list(itertools.chain(*ip_to_parts.values()))
    partition_assigned = True
    while len(rest_parts) > 0 and partition_assigned:
        partition_assigned = False
        for actor in actors:
            num_actor_parts = len(actor_to_partitions[actor])
            if min_parts_per_actor <= num_actor_parts < max_parts_per_actor:
                actor_to_partitions[actor].append(rest_parts.pop(0))
                partition_assigned = True

    if len(rest_parts) != 0:
        raise RuntimeError(
            f"There are still partitions left to assign, but no actor "
            f"has capacity for more. This is probably a bug. Please go "
            f"to https://github.com/ray-project/xgboost_ray to report it."
        )

    return actor_to_partitions
