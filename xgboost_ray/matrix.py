import math
from enum import Enum
from typing import Union, Optional, Tuple, Iterable, List

import numpy as np
import pandas as pd

import os

import ray

Data = Union[str, List[str], np.ndarray, pd.DataFrame, pd.Series]


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

    def load_data(self, num_actors: int, sharding: RayShardingMode):
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

        x_refs = {}
        y_refs = {}
        for i in range(num_actors):
            indices = _get_sharding_indices(sharding, i, num_actors, n)
            actor_x = x.iloc[indices]
            actor_y = y.iloc[indices]
            x_refs[i] = ray.put(actor_x)
            y_refs[i] = ray.put(actor_y)

        return x_refs, y_refs, n

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
                 num_actors: Optional[int] = None,
                 filetype: Optional[RayFileType] = None,
                 sharding: RayShardingMode = RayShardingMode.INTERLEAVED,
                 **kwargs):

        self.memory_node_ip = ray.services.get_node_ip_address()
        self.num_actors = num_actors
        self.sharding = sharding

        self.loader = _RayDMatrixLoader(
            data=data, label=label, filetype=filetype, **kwargs)

        self.x_ref = None
        self.y_ref = None
        self.n = None

        self.loaded = False

        if num_actors is not None:
            self.load_data(num_actors)

    def load_data(self, num_actors: Optional[int] = None):
        if not self.loaded:
            if num_actors is not None:
                self.num_actors = num_actors
            self.x_ref, self.y_ref, self.n = self.loader.load_data(
                self.num_actors, self.sharding)
            self.loaded = True

    def get_data(self, rank: int) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        self.load_data()

        x_ref = self.x_ref[rank]
        y_ref = self.y_ref[rank]

        x_df, y_df = ray.get([x_ref, y_ref])

        return x_df, y_df

    def __hash__(self):
        return hash((tuple(self.x_ref.values()), tuple(self.y_ref.values()),
                     self.n, self.sharding))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


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
