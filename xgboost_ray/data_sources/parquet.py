from typing import Any, Optional, Sequence, Iterable, Union

import pandas as pd

from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.pandas import Pandas


class Parquet(DataSource):
    """Read one or many Parquet files."""
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        return filetype == RayFileType.PARQUET

    @staticmethod
    def get_filetype(data: Any) -> Optional[RayFileType]:
        if data.endswith(".parquet"):
            return RayFileType.PARQUET
        return None

    @staticmethod
    def load_data(data: Union[str, Sequence[str]],
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs) -> pd.DataFrame:
        if isinstance(data, Iterable) and not isinstance(data, str):
            shards = []

            for i, shard in enumerate(data):
                if indices and i not in indices:
                    continue

                shard_df = pd.read_parquet(shard, **kwargs)
                shards.append(Pandas.load_data(shard_df, ignore=ignore))
            return pd.concat(shards, copy=False)
        else:
            local_df = pd.read_parquet(data, **kwargs)
            return Pandas.load_data(local_df, ignore=ignore)

    @staticmethod
    def get_n(data: Any):
        return len(list(data))
