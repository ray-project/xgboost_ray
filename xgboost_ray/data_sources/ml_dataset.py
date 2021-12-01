from typing import Any, Optional, Sequence, List, Tuple, Dict

from ray.actor import ActorHandle

import pandas as pd
from xgboost_ray.data_sources.data_source import DataSource, RayFileType

try:
    import pyarrow  # noqa: F401
    PYARROW_INSTALLED = True
except (ImportError, AttributeError):
    PYARROW_INSTALLED = False

if PYARROW_INSTALLED:
    from ray.util.data import MLDataset as MLDatasetType
else:
    MLDatasetType = None


def _assert_pyarrow_installed():
    if not PYARROW_INSTALLED:
        raise RuntimeError(
            "Tried to use MLDataset as a data source, but pyarrow is not "
            "installed. This function shouldn't have been called. "
            "\nFIX THIS by installing pyarrow: `pip install pyarrow`. "
            "\nPlease also raise an issue on our GitHub: "
            "https://github.com/ray-project/xgboost_ray as this part of "
            "the code should not have been reached.")


class MLDataset(DataSource):
    """Read from distributed Ray MLDataset.

    The Ray MLDataset is a distributed dataset based on Ray's
    `parallel iterators <https://docs.ray.io/en/master/iter.html>`_.

    Shards of the MLDataset can be stored on different nodes, making
    it suitable for distributed loading.
    """
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        if not PYARROW_INSTALLED:
            return False
        return isinstance(data, MLDatasetType)

    @staticmethod
    def load_data(data: MLDatasetType,
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs):
        _assert_pyarrow_installed()
        indices = indices or list(range(0, data.num_shards()))

        shards: List[pd.DataFrame] = [
            pd.concat(data.get_shard(i), copy=False) for i in indices
        ]

        # Concat all shards
        local_df = pd.concat(shards, copy=False)

        if ignore:
            local_df = local_df[local_df.columns.difference(ignore)]

        return local_df

    @staticmethod
    def get_n(data: MLDatasetType):
        return data.num_shards()

    @staticmethod
    def convert_to_series(data: MLDatasetType) -> pd.Series:
        _assert_pyarrow_installed()
        return super().convert_to_series(data)

    @staticmethod
    def get_actor_shards(data: MLDatasetType, actors: Sequence[ActorHandle]
                         ) -> Tuple[Any, Optional[Dict[int, Any]]]:
        _assert_pyarrow_installed()
        return super().get_actor_shards(data, actors)
