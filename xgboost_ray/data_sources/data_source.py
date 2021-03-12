from enum import Enum
from typing import Any, Optional, Sequence, Tuple

import pandas as pd


class RayFileType(Enum):
    """Enum for different file types (used for overrides)."""
    CSV = 1
    PARQUET = 2
    PETASTORM = 3


class DataSource:
    supports_central_loading = True
    supports_distributed_loading = False

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        return False

    @staticmethod
    def get_filetype(data: Any) -> Optional[RayFileType]:
        return None

    @staticmethod
    def load_data(data: Any,
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        if isinstance(data, pd.DataFrame):
            return pd.Series(data.squeeze())

        if not isinstance(data, pd.Series):
            return pd.Series(data)

        return data

    @classmethod
    def get_column(cls, data: pd.DataFrame,
                   column: Any) -> Tuple[pd.Series, Optional[str]]:
        if isinstance(column, str):
            return data[column], column
        elif column is not None:
            return cls.convert_to_series(column), None
        return column, None

    @staticmethod
    def get_n(data: Any):
        return len(list(data))
