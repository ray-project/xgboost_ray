from typing import Any, Optional, Sequence, Dict, Union, Tuple

from collections import defaultdict
import pandas as pd

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from xgboost_ray.data_sources._distributed import \
    assign_partitions_to_actors, get_actor_rank_ips
from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.object_store import ObjectStore

try:
    import modin  # noqa: F401
    from distutils.version import LooseVersion
    MODIN_INSTALLED = LooseVersion(modin.__version__) >= LooseVersion("0.9.0")
except (ImportError, AttributeError):
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

        from modin.distributed.dataframe.pandas import unwrap_partitions

        actor_rank_ips = get_actor_rank_ips(actors)

        # Get IPs and partitions
        unwrapped = unwrap_partitions(data, axis=0, get_ip=True)
        ip_objs, part_objs = zip(*unwrapped)

        # Build a table mapping from IP to list of partitions
        ip_to_parts = defaultdict(list)
        for ip, part_obj in zip(ray.get(list(ip_objs)), part_objs):
            ip_to_parts[ip].append(part_obj)

        # Modin dataframes are not serializable, so pass None here
        # as the first return value
        return None, assign_partitions_to_actors(ip_to_parts, actor_rank_ips)

    @staticmethod
    def get_n(data: Any):
        """
        For naive distributed loading we just return the number of rows
        here. Loading by shard is achieved via `get_actor_shards()`
        """
        return len(data)
