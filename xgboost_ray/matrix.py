import math
from enum import Enum
from typing import Union, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

import ray

Data = Union[str, np.ndarray, pd.DataFrame, pd.Series]


class RayFileType(Enum):
    CSV = 1
    PARQUET = 2


class RayShardingMode(Enum):
    INTERLEAVED = 1
    BATCH = 2


class _RayDMatrixLoader:
    def __init__(self,
                 data: Data,
                 label: Optional[Data] = None,
                 filetype: Optional[RayFileType] = None,
                 **kwargs):
        self.data = data
        self.label = label
        self.filetype = filetype
        self.kwargs = kwargs

        self._x_ref = None
        self._y_ref = None
        self._n = 0

        if isinstance(data, str):
            if not self.filetype:
                # Try to guess filetype from file ending
                if data.endswith(".csv"):
                    self.filetype = RayFileType.CSV
                elif data.endswith(".parquet"):
                    self.filetype = RayFileType.PARQUET
                else:
                    raise ValueError(
                        "File or stream specified as data source, but "
                        "filetype could not be detected. Please pass "
                        "the `filetype` parameter to the RayDMatrix.")

        self.load_data()

    def __hash__(self):
        return hash((
            id(self.data), id(self.label), self.filetype))

    def get_refs(self):
        return self._x_ref, self._y_ref, self._n

    def load_data(self):
        """
        Load data into memory
        """
        if self.label is not None and not isinstance(self.label, str):
            if type(self.data) != type(self.label):
                raise ValueError(
                    "If you pass a data object as label (e.g. a DataFrame), "
                    "it has to be of the same type as the main data. Got"
                    "{} for the main data and {} for the label.".format(
                        type(self.data), type(self.label)
                    ))

        if isinstance(self.data, np.ndarray):
            local_df = self._load_data_numpy()
        elif isinstance(self.data, (pd.DataFrame, pd.Series)):
            local_df = self._load_data_pandas()
        elif self.filetype == RayFileType.CSV:
            local_df = self._load_data_csv()
        elif self.filetype == RayFileType.PARQUET:
            local_df = self._load_data_parquet()
        else:
            raise ValueError(
                "Unknown data source type: {} with FileType: {}. Supported "
                "data types include pandas.DataFrame, pandas.Series, "
                "np.ndarray, and CSV/Parquet files. If you specify a file, "
                "consider passing the `filetype` argument to specify the "
                "type of the source.")

        x, y = self._split_dataframe(local_df)
        n = len(local_df)

        self._x_ref = ray.put(x)
        self._y_ref = ray.put(y)
        self._n = n

    def _split_dataframe(self, local_data: pd.DataFrame) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Split dataframe into `features`, `labels`"""
        if self.label is not None:
            if isinstance(self.label, str):
                x = local_data[local_data.columns.difference([self.label])]
                y = local_data[self.label]
            else:
                x = local_data
                y = self.label
            return x, y
        return local_data, None

    def _load_data_numpy(self):
        return pd.DataFrame(self.data)

    def _load_data_pandas(self):
        return self.data

    def _load_data_csv(self):
        if isinstance(self.data, Iterable):
            data_sources = self.data
        else:
            data_sources = [self.data]

        return pd.concat([pd.read_csv(data_source, **self.kwargs)
                          for data_source in data_sources])

    def _load_data_parquet(self):
        if isinstance(self.data, Iterable):
            return pd.concat([pd.read_parquet(data_source, **self.kwargs)
                              for data_source in self.data])
        else:
            return pd.read_parquet(self.data, **self.kwargs)


class RayDMatrix:
    def __init__(self,
                 data: Data,
                 label: Optional[Data] = None,
                 filetype: Optional[RayFileType] = None,
                 sharding: RayShardingMode = RayShardingMode.INTERLEAVED,
                 **kwargs):

        self.sharding = sharding

        loader = _RayDMatrixLoader(
            data=data,
            label=label,
            filetype=filetype,
            **kwargs)

        self.x_ref, self.y_ref, self.n = loader.get_refs()

    def get_data(self, rank: int, num_actors: int) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

        # x_df and y_df and ObjectRefs
        x_df: pd.DataFrame = ray.get(self.x_ref)
        y_df: pd.DataFrame = ray.get(self.y_ref)

        indices = self._get_sharding(rank, num_actors, self.n)

        if x_df is not None:
            x_df = x_df.iloc[indices]

        if y_df is not None:
            y_df = y_df.iloc[indices]

        return x_df, y_df

    def _get_sharding(self, rank: int, num_actors: int, n: int):
        """Return indices that belong to worker with rank `rank`"""
        if self.sharding == RayShardingMode.BATCH:
            start_index = int(math.floor(rank / num_actors) * n)
            end_index = int(math.floor(rank + 1 / num_actors) * n)
            indices = list(range(start_index, end_index))
        elif self.sharding == RayShardingMode.INTERLEAVED:
            indices = list(range(rank, n, num_actors))
        else:
            raise ValueError(f"Invalid value for `sharding` parameter: "
                             f"{self.sharding}")
        return indices

    def __hash__(self):
        return hash((self.x_ref, self.y_ref, self.n, self.sharding))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
