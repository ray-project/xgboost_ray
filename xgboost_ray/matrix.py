import math
from enum import Enum
from typing import Union, Optional, Tuple, List

import numpy as np
import pandas as pd

import ray

Data = Union[str, np.ndarray, pd.DataFrame, pd.Series]


class RayFileType(Enum):
    CSV = 1
    PARQUET = 2


class RayShardingMode(Enum):
    BATCH = 1
    INTERLEAVED = 2


@ray.remote
class _RayRemoteDMatrix:
    def __init__(self,
                 data: Data,
                 label: Optional[Data] = None,
                 filetype: Optional[RayFileType] = None,
                 sharding: RayShardingMode = RayShardingMode.INTERLEAVED):
        self.data = data
        self.label = label
        self.filetype = filetype
        self.sharding = sharding

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

        self._hash = None

    def __hash__(self):
        return hash((
            id(self.data), id(self.label), self.filetype, self.sharding))

    @ray.method(num_returns=2)
    def load_data(self, rank: int, num_actors: int) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load data shard for worker with rank `rank`.
        """
        assert rank < num_actors

        if self.label is not None and not isinstance(self.label, str):
            if type(self.data) != type(self.label):
                raise ValueError(
                    "If you pass a data object as label (e.g. a DataFrame), "
                    "it has to be of the same type as the main data. Got"
                    "{} for the main data and {} for the label.".format(
                        type(self.data), type(self.label)
                    ))

        if isinstance(self.data, np.ndarray):
            return self._load_data_numpy(rank, num_actors)
        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            return self._load_data_pandas(rank, num_actors)

    def _split_dataframe(self,
                         local_data: pd.DataFrame,
                         indices: List[int]) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if self.label is not None:
            if isinstance(self.label, str):
                x = local_data[local_data.columns.difference([self.label])]
                y = local_data[self.label]
            else:
                x = local_data
                y = self.label[indices]
            return x, y
        return local_data, None

    def _load_data_numpy(self, rank: int, num_actors: int) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        assert isinstance(self.data, np.ndarray)

        n = len(self.data)
        if self.sharding == RayShardingMode.BATCH:
            start_index = int(math.floor(rank / num_actors) * n)
            end_index = int(math.floor(rank + 1 / num_actors) * n)
            indices = list(range(start_index, end_index))
        elif self.sharding == RayShardingMode.INTERLEAVED:
            indices = list(range(rank, n, num_actors))
        else:
            raise ValueError(f"Invalid value for `sharding` parameter: "
                             f"{self.sharding}")

        local_data: pd.DataFrame = pd.DataFrame(self.data[indices])
        return self._split_dataframe(local_data, indices)

    def _load_data_pandas(self, rank: int, num_actors: int) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        assert isinstance(self.data, (pd.DataFrame, pd.Series))

        n = len(self.data)
        if self.sharding == RayShardingMode.BATCH:
            start_index = int(math.floor(rank / num_actors) * n)
            end_index = int(math.floor(rank + 1 / num_actors) * n)
            indices = list(range(start_index, end_index))
        elif self.sharding == RayShardingMode.INTERLEAVED:
            indices = list(range(rank, n, num_actors))
        else:
            raise ValueError(f"Invalid value for `sharding` parameter: "
                             f"{self.sharding}")

        local_data: pd.DataFrame = self.data.loc[indices]
        return self._split_dataframe(local_data, indices)


class RayDMatrix:
    def __init__(self,
                 data: Data,
                 label: Optional[Data] = None,
                 filetype: Optional[RayFileType] = None,
                 sharding: RayShardingMode = RayShardingMode.INTERLEAVED):

        if not ray.is_initialized():
            ray.init()

        self._dmatrix = _RayRemoteDMatrix.remote(
            data=data,
            label=label,
            filetype=filetype,
            sharding=sharding)

    def load_data(self, rank: int, num_actors: int) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        return self._dmatrix.load_data.remote(
            rank=rank,
            num_actors=num_actors)

    def __hash__(self):
        return ray.get(self._dmatrix.__hash__.remote())

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
