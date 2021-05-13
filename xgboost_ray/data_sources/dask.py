from collections import defaultdict
from typing import Any, Optional, Sequence, Dict, Union, Tuple

import pandas as pd

import ray
from ray.actor import ActorHandle

from xgboost_ray.data_sources._distributed import \
    assign_partitions_to_actors, get_actor_rank_ips
from xgboost_ray.data_sources.data_source import DataSource, RayFileType

try:
    import dask  # noqa: F401
    from ray.util.dask import ray_dask_get
    DASK_INSTALLED = True
except ImportError:
    DASK_INSTALLED = False


def _assert_dask_installed():
    if not DASK_INSTALLED:
        raise RuntimeError(
            "Tried to use Dask as a data source, but dask is not "
            "installed. This function shouldn't have been called. "
            "\nFIX THIS by installing dask: `pip install dask`. "
            "\nPlease also raise an issue on our GitHub: "
            "https://github.com/ray-project/xgboost_ray as this part of "
            "the code should not have been reached.")


class Dask(DataSource):
    """Read from distributed Dask dataframe.

    A `Dask dataframe <https://docs.dask.org/en/latest/dataframe.html>`_
    is a distributed drop-in replacement for pandas.

    Dask dataframes are stored on multiple actors, making them
    suitable for distributed loading.
    """
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        if not DASK_INSTALLED:
            return False
        from dask.dataframe import DataFrame as DaskDataFrame, \
            Series as DaskSeries

        return isinstance(data, (DaskDataFrame, DaskSeries))

    @staticmethod
    def load_data(
            data: Any,  # dask.pandas.DataFrame
            ignore: Optional[Sequence[str]] = None,
            indices: Optional[Union[Sequence[int], Sequence[int]]] = None,
            **kwargs) -> pd.DataFrame:
        _assert_dask_installed()

        import dask.dataframe as dd

        if indices is not None and len(indices) > 0 and isinstance(
                indices[0], Tuple):
            # We got a list of partition IDs belonging to Dask partitions
            return dd.concat(
                [data.partitions[i] for (i, ) in indices]).compute()

        # Dask does not support iloc() for row selection, so we have to
        # compute a local pandas dataframe first
        local_df = data.compute()

        if indices:
            local_df = local_df.iloc[indices]

        if ignore:
            local_df = local_df[local_df.columns.difference(ignore)]

        return local_df

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        _assert_dask_installed()
        from dask.dataframe import DataFrame as DaskDataFrame, \
            Series as DaskSeries
        from dask.array import Array as DaskArray

        if isinstance(data, DaskDataFrame):
            return pd.Series(data.compute().squeeze())
        elif isinstance(data, DaskSeries):
            return data.compute()
        elif isinstance(data, DaskArray):
            return pd.Series(data.compute())

        return DataSource.convert_to_series(data)

    @staticmethod
    def get_actor_shards(
            data: Any,  # dask.dataframe.DataFrame
            actors: Sequence[ActorHandle]) -> \
            Tuple[Any, Optional[Dict[int, Any]]]:
        _assert_dask_installed()

        actor_rank_ips = get_actor_rank_ips(actors)

        # Get IPs and partitions
        ip_to_parts = get_ip_to_parts(data)

        return data, assign_partitions_to_actors(ip_to_parts, actor_rank_ips)

    @staticmethod
    def get_n(data: Any):
        """
        For naive distributed loading we just return the number of rows
        here. Loading by shard is achieved via `get_actor_shards()`
        """
        return len(data)


def get_ip_to_parts(data: Any) -> Dict[int, Sequence[Any]]:
    persisted = data.persist(scheduler=ray_dask_get)
    name = persisted._name

    node_ids_to_node = {node["NodeID"]: node for node in ray.state.nodes()}

    # This is a hacky way to get the partition node IDs, and it's not
    # 100% accurate as the map task could get scheduled on a different node
    # (though Ray tries to keep locality). We need to use that until
    # ray.state.objects() or something like it is available again.
    partition_locations_df = persisted.map_partitions(lambda df: pd.DataFrame(
        [ray.get_runtime_context().node_id.hex()])).compute()
    partition_locations = [
        partition_locations_df[0].iloc[i]
        for i in range(partition_locations_df.size)
    ]

    ip_to_parts = defaultdict(list)
    for (obj_name,
         pid), obj_ref in dask.base.collections_to_dsk([persisted]).items():
        assert obj_name == name

        if isinstance(obj_ref, ray.ObjectRef):
            node_id = partition_locations[pid]
            node = node_ids_to_node.get(node_id, {})
            ip = node.get("NodeManagerAddress", "_no_ip")
        else:
            ip = "_no_ip"

        # Pass tuples here (integers can be misinterpreted as row numbers)
        ip_to_parts[ip].append((pid, ))

    return ip_to_parts
