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
    """XGBoost on Ray DMatrix class.

    This is the data object that the training and prediction functions
    expect. This wrapper manages distributed data by sharding the data for the
    workers and storing the shards in the object store.

    If this class is called without the ``num_actors`` argument, it will
    be lazy loaded. Thus, it will return immediately and only load the data
    and store it in the Ray object store after ``load_data(num_actors)`` or
    ``get_data(rank, num_actors)`` is called.

    If this class is instantiated with the ``num_actors`` argument, it will
    directly load the data and store them in the object store. If this should
    be deferred, pass ``lazy=True`` as an argument.

    Loading the data will store it in the Ray object store. This object then
    stores references to the data shards in the Ray object store. Actors
    can request these shards with the ``get_data(rank)`` method, returning
    dataframes according to the actor rank.

    The total number of actors has to remain constant and cannot be changed
    once it has been set.

    Args:
        data: Data object. Can be a pandas dataframe, pandas series,
            numpy array, string pointing to a csv or parquet file, or
            list of strings pointing to csv or parquet files.
        label: Optional label object. Can be a pandas series,
            numpy array, string pointing to a csv or parquet file, or
            a string indicating the column of the data dataframe that
            contains the label. If this is not a string it must be of the
            same type as the data argument.
        num_actors: Number of actors to shard this data for. If this is
            not None, data will be loaded and stored into the object store
            after initialization. If this is None, it will be set by
            the ``xgboost_ray.train()`` function, and it will be loaded and
            stored in the object store then. Defaults to None (
        filetype (RayFileType): Type of data to read. This is disregarded if
            a data object like a pandas dataframe is passed as the ``data``
            argument. For filenames, the filetype is automaticlly detected
            via the file name (e.g. ``.csv`` will be detected as
            ``RayFileType.CSV``). Passing this argument will overwrite the
            detected filename. If the filename cannot be determined from
            the ``data`` object, passing this is mandatory. Defaults to
            ``None`` (auto detection).
        sharding (RayShardingMode): How to shard the data for different
            workers. ``RayShardingMode.INTERLEAVED`` will divide the data
            per row, i.e. every i-th row will be passed to the first worker,
            every (i+1)th row to the second worker, etc.
            ``RayShardingMode.BATCH`` will divide the data in batches, i.e.
            the first 0-(m-1) rows will be passed to the first worker, the
            m-(2m-1) rows to the second worker, etc. Defaults to
            ``RayShardingMode.INTERLEAVED``.
        lazy (bool): If ``num_actors`` is passed, setting this to ``True``
            will defer data loading and storing until ``load_data()`` or
            ``get_data()`` is called. Defaults to ``False``.
        **kwargs: Keyword arguments will be passed to the data loading
            function. For instance, with ``RayFileType.PARQUET``, these
            arguments will be passed to ``pandas.read_parquet()``.


    .. code-block:: python

        from xgboost_ray import RayDMatrix, RayFileType

        files = ["data_one.parquet", "data_two.parquet"]

        columns = ["feature_1", "feature_2", "label_column"]

        dtrain = RayDMatrix(
            files,
            num_actors=4,  # Will shard the data for four workers
            label="label_column",  # Will select this column as the label
            columns=columns,  # Will be passed to `pandas.read_parquet()`
            filetype=RayFileType.PARQUET)

    """

    def __init__(self,
                 data: Data,
                 label: Optional[Data] = None,
                 num_actors: Optional[int] = None,
                 filetype: Optional[RayFileType] = None,
                 sharding: RayShardingMode = RayShardingMode.INTERLEAVED,
                 lazy: bool = False,
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

        if num_actors is not None and not lazy:
            self.load_data(num_actors)

    def load_data(self, num_actors: Optional[int] = None):
        if not self.loaded:
            if num_actors is not None:
                if self.num_actors is not None:
                    raise ValueError(
                        f"The `RayDMatrix` was initialized or `load_data()`"
                        f"has been called with a different numbers of"
                        f"`actors`. Existing value: {self.num_actors}. "
                        f"Current value: {num_actors}."
                        f"\nFIX THIS by not instantiating the matrix with "
                        f"`num_actors` and making sure calls to `load_data()` "
                        f"or `get_data()` use the same numbers of actors "
                        f"at each call.")
                self.num_actors = num_actors
            if self.num_actors is None:
                raise ValueError(
                    "Trying to load data for `RayDMatrix` object, but "
                    "`num_actors` is not set."
                    "\nFIX THIS by passing `num_actors` on instantiation "
                    "of the `RayDMatrix` or when calling `load_data()`.")
            self.x_ref, self.y_ref, self.n = self.loader.load_data(
                self.num_actors, self.sharding)
            self.loaded = True

    def get_data(self, rank: int, num_actors: Optional[int] = None) -> \
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        self.load_data(num_actors=num_actors)

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


def combine_data(sharding: RayShardingMode, data: Iterable) -> np.ndarray:
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
