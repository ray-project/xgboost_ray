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
    import ray.data.dataset  # noqa: F401
    RAY_DATASET_AVAILABLE = True
except (ImportError, AttributeError):
    RAY_DATASET_AVAILABLE = False


def _assert_ray_data_available():
    if not RAY_DATASET_AVAILABLE:
        raise RuntimeError(
            "Tried to use Ray datasets as a data source, but your version "
            "of Ray does not support it. "
            "\nFIX THIS by upgrading Ray: `pip install -U ray`. "
            "\nPlease also raise an issue on our GitHub: "
            "https://github.com/ray-project/xgboost_ray as this part of "
            "the code should not have been reached.")


class RayDataset(DataSource):
    """Read from distributed Ray dataset."""
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        if not RAY_DATASET_AVAILABLE:
            return False

        return isinstance(data, ray.data.dataset.Dataset)

    @staticmethod
    def load_data(
            data: Any,  # ray.data.dataset.Dataset
            ignore: Optional[Sequence[str]] = None,
            indices: Optional[Union[Sequence[int], Sequence[
                ObjectRef]]] = None,
            **kwargs) -> pd.DataFrame:
        _assert_ray_data_available()

        if indices is not None and len(indices) > 0 and isinstance(
                indices[0], ObjectRef):
            # We got a list of ObjectRefs belonging to Ray dataset partitions
            return ObjectStore.load_data(
                data=indices, indices=None, ignore=ignore)

        if hasattr(data, "to_pandas_refs"):
            obj_refs = data.to_pandas_refs()
        else:
            # Legacy API
            obj_refs = data.to_pandas()

        ray.wait(obj_refs)
        return ObjectStore.load_data(obj_refs, ignore=ignore, indices=indices)

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        _assert_ray_data_available()

        obj_refs = data.to_pandas()
        return ObjectStore.convert_to_series(obj_refs)

    @staticmethod
    def get_actor_shards(
            data: Any,  # ray.data.dataset.Dataset
            actors: Sequence[ActorHandle]) -> \
            Tuple[Any, Optional[Dict[int, Any]]]:
        _assert_ray_data_available()

        actor_rank_ips = get_actor_rank_ips(actors)

        # Map node IDs to IP
        node_id_to_ip = {
            node["NodeID"]: node["NodeManagerAddress"]
            for node in ray.nodes()
        }

        # Get object store locations
        if hasattr(data, "to_pandas_refs"):
            obj_refs = data.to_pandas_refs()
        else:
            # Legacy API
            obj_refs = data.to_pandas()
        ray.wait(obj_refs)

        ip_to_parts = defaultdict(list)
        for part_obj, location in ray.experimental.get_object_locations(
                obj_refs).items():
            if len(location["node_ids"]) == 0:
                node_id = None
            else:
                node_id = location["node_ids"][0]

            ip = node_id_to_ip.get(node_id, None)
            ip_to_parts[ip].append(part_obj)

        # Ray datasets should not be serialized
        return None, assign_partitions_to_actors(ip_to_parts, actor_rank_ips)

    @staticmethod
    def get_n(data: Any):
        """
        Return number of distributed blocks.
        """
        return data.num_blocks()
