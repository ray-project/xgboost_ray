from typing import Any, Optional, Sequence, List

import pandas as pd
from ray.util.data import MLDataset as MLDatasetType
from xgboost_ray.data_sources.data_source import DataSource, RayFileType


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
        return isinstance(data, MLDatasetType)

    @staticmethod
    def load_data(data: MLDatasetType,
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs):
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
