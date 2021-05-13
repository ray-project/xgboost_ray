from typing import Any, Optional, Sequence

import pandas as pd

from xgboost_ray.data_sources.data_source import DataSource, RayFileType


class Pandas(DataSource):
    """Read from pandas dataframes and series."""

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        return isinstance(data, (pd.DataFrame, pd.Series))

    @staticmethod
    def load_data(data: Any,
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs) -> pd.DataFrame:
        local_df = data

        if ignore:
            local_df = local_df[local_df.columns.difference(ignore)]

        if indices:
            return local_df.iloc[indices]

        return local_df
