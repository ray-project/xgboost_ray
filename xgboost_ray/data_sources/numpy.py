from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.pandas import Pandas


class Numpy(DataSource):
    """Read from numpy arrays."""

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        return isinstance(data, np.ndarray)

    @staticmethod
    def load_data(data: np.ndarray,
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs) -> pd.DataFrame:
        local_df = pd.DataFrame(
            data, columns=[f"f{i}" for i in range(data.shape[1])])
        return Pandas.load_data(local_df, ignore=ignore, indices=indices)
