import math
from enum import Enum
from typing import Union, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

import os

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
                        "filetype could not be detected. "
                        "\nFIX THIS by passing] "
                        "the `filetype` parameter to the RayDMatrix. Use the "
                        "`RayFileType` enum for this.")

    def __hash__(self):
        return hash((id(self.data), id(self.label), self.filetype))

    def load_data(self):
        """
        Load data into memory
        """
        if not ray.is_initialized():
            ray.init()

        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]

        if self.label is not None and not isinstance(self.label, str) and \
           not (isinstance(self.data, pd.DataFrame) and
                isinstance(self.label, pd.Series)):
            if type(self.data) != type(self.label):  # noqa: E721
                raise ValueError(
                    "The passed `data` and `label` types are not compatible."
                    "\nFIX THIS by passing the same types to the "
                    "`RayDMatrix` - e.g. a `pandas.DataFrame` as `data` "
                    "and `label`. The `label` can always be a string. Got "
                    "{} for the main data and {} for the label.".format(
                        type(self.data), type(self.label)))

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
                "Unknown data source type: {} with FileType: {}."
                "\nFIX THIS by passing a supported data type. Supported "
                "data types include pandas.DataFrame, pandas.Series, "
                "np.ndarray, and CSV/Parquet file paths. If you specify a "
                "file, path, consider passing the `filetype` argument to "
                "specify the type of the source. Use the `RayFileType` "
                "enum for that.")

        x, y = self._split_dataframe(local_df)
        n = len(local_df)

        return ray.put(x), ray.put(y), n

    def _split_dataframe(self, local_data: pd.DataFrame) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Split dataframe into `features`, `labels`"""
        if self.label is not None:
            if isinstance(self.label, str):
                x = local_data[local_data.columns.difference([self.label])]
                y = local_data[self.label]
            else:
                x = local_data
                if not isinstance(self.label, pd.DataFrame):
                    y = pd.DataFrame(self.label)
                else:
                    y = self.label
            return x, y
        return local_data, None

    def _load_data_numpy(self):
        return pd.DataFrame(
            self.data, columns=[f"f{i}" for i in range(self.data.shape[1])])

    def _load_data_pandas(self):
        return self.data

    def _load_data_csv(self):
        if isinstance(self.data, Iterable) and not isinstance(self.data, str):
            return pd.concat([
                pd.read_csv(data_source, **self.kwargs)
                for data_source in self.data
            ])
        else:
            return pd.read_csv(self.data, **self.kwargs)

    def _load_data_parquet(self):
        if isinstance(self.data, Iterable) and not isinstance(self.data, str):
            return pd.concat([
                pd.read_parquet(data_source, **self.kwargs)
                for data_source in self.data
            ])
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
        self.memory_node_ip = ray.services.get_node_ip_address()

        loader = _RayDMatrixLoader(
            data=data, label=label, filetype=filetype, **kwargs)

        self.x_ref, self.y_ref, self.n = loader.load_data()

    def get_data(self, rank: int, num_actors: int) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

        x_ref, y_ref = get_data.options(
            num_cpus=0,
            resources={
                f"node:{self.memory_node_ip}": 0.1
            },
            num_returns=2).remote(self.x_ref, self.y_ref, self.sharding, rank,
                                  num_actors)

        x_df, y_df = ray.get([x_ref, y_ref])

        return x_df, y_df

    def __hash__(self):
        return hash((self.x_ref, self.y_ref, self.n, self.sharding))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


@ray.remote
def get_data(x_df, y_df, sharding: RayShardingMode, rank: int,
             num_actors: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

    n = len(x_df)
    indices = _get_sharding_indices(sharding, rank, num_actors, n)

    if x_df is not None:
        x_df = x_df.iloc[indices]

    if y_df is not None:
        y_df = y_df.iloc[indices]

    return x_df, y_df


def _get_sharding_indices(sharding: RayShardingMode, rank: int,
                          num_actors: int, n: int):
    """Return indices that belong to worker with rank `rank`"""
    if sharding == RayShardingMode.BATCH:
        start_index = int(math.floor(rank / num_actors) * n)
        end_index = int(math.floor(rank + 1 / num_actors) * n)
        indices = list(range(start_index, end_index))
    elif sharding == RayShardingMode.INTERLEAVED:
        indices = list(range(rank, n, num_actors))
    else:
        raise ValueError(f"Invalid value for `sharding` parameter: "
                         f"{sharding}"
                         f"\nFIX THIS by passing any item of the "
                         f"`RayShardingMode` enum, for instance "
                         f"`RayShardingMode.BATCH`.")
    return indices


def combine_data(sharding: RayShardingMode, data: Iterable):
    if sharding == RayShardingMode.BATCH:
        np.ravel(data)
    elif sharding == RayShardingMode.INTERLEAVED:
        # Sometimes the lengths are off by 1 for uneven divisions
        min_len = min([len(d) for d in data])
        res = np.ravel(np.column_stack([d[0:min_len] for d in data]))
        # Append these here
        res = np.concatenate([res] +
                             [d[min_len:] for d in data if len(d) > min_len])
        return res
    else:
        raise ValueError(f"Invalid value for `sharding` parameter: "
                         f"{sharding}"
                         f"\nFIX THIS by passing any item of the "
                         f"`RayShardingMode` enum, for instance "
                         f"`RayShardingMode.BATCH`.")
