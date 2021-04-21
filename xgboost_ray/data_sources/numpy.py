from typing import Any, Optional, Sequence, List

import numpy as np
import pandas as pd
import xgboost as xgb

from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.pandas import Pandas


class Numpy(DataSource):
    """Read from numpy arrays."""

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        return isinstance(data, np.ndarray)

    @staticmethod
    def update_feature_names(matrix: xgb.DMatrix,
                             feature_names: Optional[List[str]]):
        # Potentially unset feature names
        matrix.feature_names = feature_names

    @staticmethod
    def load_data(data: np.ndarray,
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs) -> pd.DataFrame:
        local_df = pd.DataFrame(
            data, columns=[f"f{i}" for i in range(data.shape[1])])
        return Pandas.load_data(local_df, ignore=ignore, indices=indices)
