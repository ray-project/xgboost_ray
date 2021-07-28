from typing import Any, Optional, Sequence

import pandas as pd

import ray
from ray import ObjectRef

from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.pandas import Pandas


class ObjectStore(DataSource):
    """Read pandas dataframes and series from ray object store."""

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        if isinstance(data, Sequence):
            return all(isinstance(d, ObjectRef) for d in data)
        return isinstance(data, ObjectRef)

    @staticmethod
    def load_data(data: Sequence[ObjectRef],
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs) -> pd.DataFrame:
        if indices is not None:
            data = [data[i] for i in indices]

        local_df = ray.get(data)

        return Pandas.load_data(pd.concat(local_df, copy=False), ignore=ignore)

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        if isinstance(data, ObjectRef):
            data = ray.get(data)
        else:
            data = pd.concat(ray.get(data), copy=False)
        return DataSource.convert_to_series(data)
