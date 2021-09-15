from typing import Any, Optional, Sequence, Dict, Tuple

from collections import defaultdict
import pandas as pd
import numpy as np

from ray import ObjectRef
from ray.actor import ActorHandle

from xgboost_ray.data_sources._distributed import \
    assign_partitions_to_actors, get_actor_rank_ips
from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.pandas import Pandas
from xgboost_ray.data_sources.numpy import Numpy


class Partitioned(DataSource):
    """Read from distributed data structure implementing __partitioned__.

    __partitioned__ provides meta data about how the data is partitioned and
    distributed across several compute nodes, making supporting objects them
    suitable for distributed loading.

    Also see the __partitioned__ spec:
    https://github.com/IntelPython/DPPY-Spec/blob/draft/partitioned/Partitioned.md
    """
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        return hasattr(data, "__partitioned__")

    @staticmethod
    def load_data(
            data: Any,  # __partitioned__ dict
            ignore: Optional[Sequence[str]] = None,
            indices: Optional[Sequence[ObjectRef]] = None,
            **kwargs) -> pd.DataFrame:

        assert isinstance(data, dict), "Expected __partitioned__ dict"
        _get = data["get"]

        if indices is None or len(indices) == 0:
            tiling = data["partition_tiling"]
            ndims = len(tiling)
            # we need tuples to access partitions in the right order
            pos_suffix = (0, ) * (ndims - 1)
            parts = data["partitions"]
            # get the full data, e.g. all shards/partitions
            local_df = [
                _get(parts[(i, ) + pos_suffix]["data"])
                for i in range(tiling[0])
            ]
        else:
            # here we got a list of futures for partitions
            local_df = _get(indices)

        if isinstance(local_df[0], pd.DataFrame):
            return Pandas.load_data(
                pd.concat(local_df, copy=False), ignore=ignore)
        else:
            return Numpy.load_data(np.concatenate(local_df), ignore=ignore)

    @staticmethod
    def get_actor_shards(
            data: Any,  # partitioned.pandas.DataFrame
            actors: Sequence[ActorHandle]) -> \
            Tuple[Any, Optional[Dict[int, Any]]]:
        assert hasattr(data, "__partitioned__")

        actor_rank_ips = get_actor_rank_ips(actors)

        # Get accessor func and partitions
        parted = data.__partitioned__
        parts = parted["partitions"]
        tiling = parted["partition_tiling"]
        ndims = len(tiling)
        if ndims < 1 or ndims > 2 or any(tiling[x] != 1
                                         for x in range(1, ndims)):
            raise RuntimeError(
                "Only row-wise partitionings of 1d/2d structures supported.")

        # Now build a table mapping from IP to list of partitions
        ip_to_parts = defaultdict(lambda: [])
        # we need tuples to access partitions in the right order
        pos_suffix = (0, ) * (ndims - 1)
        for i in range(tiling[0]):
            part = parts[(i, ) + pos_suffix]  # this works for 1d and 2d
            ip_to_parts[part["location"][0]].append(part["data"])
        # __partitioned__ is serializable, so pass it here
        # as the first return value
        ret = parted, assign_partitions_to_actors(ip_to_parts, actor_rank_ips)
        return ret

    @staticmethod
    def get_n(data: Any):
        """Get length of data source partitions for sharding."""
        return data.__partitioned__["shape"][0]
