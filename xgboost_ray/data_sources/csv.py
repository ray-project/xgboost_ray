from typing import Any, Optional, Sequence, Iterable, Union

import pandas as pd

from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.pandas import Pandas


class CSV(DataSource):
    """Read one or many CSV files."""
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        return filetype == RayFileType.CSV

    @staticmethod
    def get_filetype(data: Any) -> Optional[RayFileType]:
        if data.endswith(".csv") or data.endswith("csv.gz"):
            return RayFileType.CSV
        return None

    @staticmethod
    def load_data(data: Union[str, Sequence[str]],
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs):
        if isinstance(data, Iterable) and not isinstance(data, str):
            shards = []

            for i, shard in enumerate(data):
                if indices and i not in indices:
                    continue
                shard_df = pd.read_csv(shard, **kwargs)
                shards.append(Pandas.load_data(shard_df, ignore=ignore))
            return pd.concat(shards, copy=False)
        else:
            local_df = pd.read_csv(data, **kwargs)
            return Pandas.load_data(local_df, ignore=ignore)

    @staticmethod
    def get_n(data: Any):
        return len(list(data))
