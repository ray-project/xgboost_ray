import glob
import math
import uuid
from enum import Enum
from typing import Union, Optional, Tuple, Iterable, List, Dict, Sequence, \
    Callable

try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np
import pandas as pd

import os

import ray

from xgboost_ray.util import Unavailable

try:
    from ray.util.data import MLDataset
except ImportError:
    MLDataset = Unavailable

try:
    import modin  # noqa: F401
    MODIN_INSTALLED = True
except ImportError:
    MODIN_INSTALLED = False

from xgboost.core import DataIter

Data = Union[str, List[str], np.ndarray, pd.DataFrame, pd.Series, MLDataset]


def concat_dataframes(dfs: List[Optional[pd.DataFrame]]):
    if any(df is None for df in dfs):
        return None
    return pd.concat(dfs, ignore_index=True, copy=False)


def _is_modin_df(df):
    if not MODIN_INSTALLED:
        return False
    from modin.pandas.dataframe import DataFrame as ModinDataFrame
    return isinstance(df, ModinDataFrame)


def _is_modin_series(df):
    if not MODIN_INSTALLED:
        return False
    from modin.pandas.dataframe import Series as ModinSeries
    return isinstance(df, ModinSeries)


class RayFileType(Enum):
    """Enum for different file types (used for overrides)."""
    CSV = 1
    PARQUET = 2


class RayShardingMode(Enum):
    """Enum for different modes of sharding the data.

    ``RayShardingMode.INTERLEAVED`` will divide the data
    per row, i.e. every i-th row will be passed to the first worker,
    every (i+1)th row to the second worker, etc.

    ``RayShardingMode.BATCH`` will divide the data in batches, i.e.
    the first 0-(m-1) rows will be passed to the first worker, the
    m-(2m-1) rows to the second worker, etc.
    """
    INTERLEAVED = 1
    BATCH = 2


class RayDataIter(DataIter):
    def __init__(
            self,
            data: List[Data],
            label: List[Optional[Data]],
            missing: Optional[float],
            weight: List[Optional[Data]],
            base_margin: List[Optional[Data]],
            label_lower_bound: List[Optional[Data]],
            label_upper_bound: List[Optional[Data]],
            feature_names: Optional[List[str]],
            feature_types: Optional[List[np.dtype]],
    ):
        super(RayDataIter, self).__init__()

        assert cp is not None

        self._data = data
        self._label = label
        self._missing = missing
        self._weight = weight
        self._base_margin = base_margin
        self._label_lower_bound = label_lower_bound
        self._label_upper_bound = label_upper_bound
        self._feature_names = feature_names
        self._feature_types = feature_types

        self._iter = 0

    def __len__(self):
        return sum(len(shard) for shard in self._data)

    def reset(self):
        self._iter = 0

    def _prop(self, ref):
        if ref is None:
            return None
        item = ref[self._iter]
        if item is None:
            return None
        if not isinstance(item, cp.ndarray):
            item = cp.array(item.values)
        return item

    def next(self, input_data: Callable):
        if self._iter >= len(self._data):
            return 0
        input_data(
            data=self._prop(self._data),
            label=self._prop(self._label),
            weight=self._prop(self._weight),
            group=None,
            label_lower_bound=self._prop(self._label_lower_bound),
            label_upper_bound=self._prop(self._label_upper_bound),
            feature_names=self._feature_names,
            feature_types=self._feature_types)
        self._iter += 1
        return 1


class _RayDMatrixLoader:
    def __init__(self,
                 data: Data,
                 label: Optional[Data] = None,
                 missing: Optional[float] = None,
                 weight: Optional[Data] = None,
                 base_margin: Optional[Data] = None,
                 label_lower_bound: Optional[Data] = None,
                 label_upper_bound: Optional[Data] = None,
                 feature_names: Optional[List[str]] = None,
                 feature_types: Optional[List[np.dtype]] = None,
                 filetype: Optional[RayFileType] = None,
                 ignore: Optional[List[str]] = None,
                 **kwargs):
        self.data = data
        self.label = label
        self.missing = missing
        self.weight = weight
        self.base_margin = base_margin
        self.label_lower_bound = label_lower_bound
        self.label_upper_bound = label_upper_bound
        self.feature_names = feature_names
        self.feature_types = feature_types

        self.filetype = filetype
        self.ignore = ignore
        self.kwargs = kwargs

        check = None
        if isinstance(data, str):
            check = data
        elif isinstance(data, Sequence) and isinstance(data[0], str):
            check = data[0]

        if check is not None:
            if not self.filetype:
                # Try to guess filetype from file ending
                if check.endswith(".csv") or check.endswith("csv.gz"):
                    self.filetype = RayFileType.CSV
                elif check.endswith(".parquet"):
                    self.filetype = RayFileType.PARQUET
                else:
                    raise ValueError(
                        f"File or stream ({check}) specified as data source, "
                        "but filetype could not be detected. "
                        "\nFIX THIS by passing "
                        "the `filetype` parameter to the RayDMatrix. Use the "
                        "`RayFileType` enum for this.")

    def _get_column(self, local_data: pd.DataFrame,
                    column: Data) -> Tuple[pd.Series, Optional[str]]:
        if isinstance(column, str):
            return local_data[column], column
        elif column is not None:
            if isinstance(column, pd.DataFrame):
                col = pd.Series(column.squeeze())
            elif _is_modin_df(column):
                col = pd.Series(column._to_pandas().squeeze())
            elif _is_modin_series(column):
                col = column._to_pandas()
            elif not isinstance(column, pd.Series):
                col = pd.Series(column)
            else:
                col = column
            return col, None
        return column, None

    def _split_dataframe(self, local_data: pd.DataFrame) -> \
            Tuple[pd.DataFrame,
                  Optional[pd.Series],
                  Optional[pd.Series],
                  Optional[pd.Series],
                  Optional[pd.Series],
                  Optional[pd.Series]]:
        """
        Split dataframe into

        `features`, `labels`, `weight`, `base_margin`, `label_lower_bound`,
        `label_upper_bound`

        """
        exclude_cols: List[str] = []  # Exclude these columns from `x`

        label, exclude = self._get_column(local_data, self.label)
        if exclude:
            exclude_cols.append(exclude)

        weight, exclude = self._get_column(local_data, self.weight)
        if exclude:
            exclude_cols.append(exclude)

        base_margin, exclude = self._get_column(local_data, self.base_margin)
        if exclude:
            exclude_cols.append(exclude)

        label_lower_bound, exclude = self._get_column(local_data,
                                                      self.label_lower_bound)
        if exclude:
            exclude_cols.append(exclude)

        label_upper_bound, exclude = self._get_column(local_data,
                                                      self.label_upper_bound)
        if exclude:
            exclude_cols.append(exclude)

        x = local_data
        if exclude_cols:
            x = x[x.columns.difference(exclude_cols)]

        return x, label, weight, base_margin, label_lower_bound, \
            label_upper_bound

    def _load_data_numpy(self, data: Data):
        local_df = pd.DataFrame(
            data, columns=[f"f{i}" for i in range(data.shape[1])])
        return self._load_data_pandas(local_df)

    def _load_data_pandas(self, data: Data):
        local_df = data

        if self.ignore:
            local_df = local_df[local_df.columns.difference(self.ignore)]

        x, y, w, b, ll, lu = self._split_dataframe(local_df)
        return x, y, w, b, ll, lu

    def _load_data_modin(self, data: Data,
                         indices: Optional[List[int]] = None):
        local_df = data
        if indices:
            local_df = local_df.iloc(indices)

        local_df = local_df._to_pandas()

        if self.ignore:
            local_df = local_df[local_df.columns.difference(self.ignore)]

        x, y, w, b, ll, lu = self._split_dataframe(local_df)
        return x, y, w, b, ll, lu

    def _load_data_csv(self, data: Data):
        if isinstance(data, Iterable) and not isinstance(data, str):
            x_s, y_s, w_s, b_s, ll_s, lu_s = [], [], [], [], [], []
            for shard in data:
                shard_df = pd.read_csv(shard, **self.kwargs)
                shard_tuple = self._load_data_pandas(shard_df)
                for i, s in enumerate([x_s, y_s, w_s, b_s, ll_s, lu_s]):
                    s.append(shard_tuple[i])
            return x_s, y_s, w_s, b_s, ll_s, lu_s
        else:
            local_df = pd.read_csv(data, **self.kwargs)
            return self._load_data_pandas(local_df)

    def _load_data_parquet(self, data: Data):
        if isinstance(data, Iterable) and not isinstance(data, str):
            x_s, y_s, w_s, b_s, ll_s, lu_s = [], [], [], [], [], []
            for shard in data:
                shard_df = pd.read_parquet(shard, **self.kwargs)
                shard_tuple = self._load_data_pandas(shard_df)
                for i, s in enumerate([x_s, y_s, w_s, b_s, ll_s, lu_s]):
                    s.append(shard_tuple[i])
            return x_s, y_s, w_s, b_s, ll_s, lu_s
        else:
            local_df = pd.read_parquet(data, **self.kwargs)
            return self._load_data_pandas(local_df)

    def _load_data_ml_dataset(self, data: MLDataset, indices: List[int]):
        # Shards can have multiple items, all of which will be DataFrames
        shards: List[pd.DataFrame] = [
            pd.concat(data.get_shard(i), copy=False) for i in indices
        ]

        # Concat all shards
        local_df = pd.concat(shards, copy=False)

        if self.ignore:
            local_df = local_df[local_df.columns.difference(self.ignore)]

        x, y, w, b, ll, lu = self._split_dataframe(local_df)
        return x, y, w, b, ll, lu

    def load_data(self,
                  num_actors: int,
                  sharding: RayShardingMode,
                  rank: Optional[int] = None) -> Tuple[Dict, int]:
        raise NotImplementedError


class _CentralRayDMatrixLoader(_RayDMatrixLoader):
    """Load full dataset from a central location and put into object store"""

    def load_data(self,
                  num_actors: int,
                  sharding: RayShardingMode,
                  rank: Optional[int] = None) -> Tuple[Dict, int]:
        """
        Load data into memory
        """
        if not ray.is_initialized():
            ray.init()

        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]

        if self.label is not None and not isinstance(self.label, str) and \
           not (isinstance(self.data, pd.DataFrame) and
                isinstance(self.label, pd.Series)) and \
           not (_is_modin_df(self.data) and _is_modin_series(self.label)):
            if type(self.data) != type(self.label):  # noqa: E721
                raise ValueError(
                    "The passed `data` and `label` types are not compatible."
                    "\nFIX THIS by passing the same types to the "
                    "`RayDMatrix` - e.g. a `pandas.DataFrame` as `data` "
                    "and `label`. The `label` can always be a string. Got "
                    "{} for the main data and {} for the label.".format(
                        type(self.data), type(self.label)))

        if isinstance(self.data, np.ndarray):
            x, y, w, b, ll, lu = self._load_data_numpy(self.data)
        elif isinstance(self.data, (pd.DataFrame, pd.Series)):
            x, y, w, b, ll, lu = self._load_data_pandas(self.data)
        elif _is_modin_df(self.data) or _is_modin_series(self.data):
            x, y, w, b, ll, lu = self._load_data_modin(self.data)
        elif isinstance(self.data, MLDataset):
            x, y, w, b, ll, lu = self._load_data_ml_dataset(
                self.data, indices=list(range(0, self.data.num_shards())))
        elif self.filetype == RayFileType.CSV:
            x, y, w, b, ll, lu = self._load_data_csv(self.data)
        elif self.filetype == RayFileType.PARQUET:
            x, y, w, b, ll, lu = self._load_data_parquet(self.data)
        else:
            raise ValueError(
                "Unknown data source type: {} with FileType: {}."
                "\nFIX THIS by passing a supported data type. Supported "
                "data types include pandas.DataFrame, pandas.Series, "
                "np.ndarray, and CSV/Parquet file paths. If you specify a "
                "file, path, consider passing the `filetype` argument to "
                "specify the type of the source. Use the `RayFileType` "
                "enum for that.".format(type(self.data), self.filetype))

        if isinstance(x, list):
            n = sum(len(a) for a in x)
        else:
            n = len(x)

        refs = {}
        for i in range(num_actors):
            indices = _get_sharding_indices(sharding, i, num_actors, n)
            actor_refs = {
                "data": ray.put(x.iloc[indices]),
                "label": ray.put(y.iloc[indices] if y is not None else None),
                "weight": ray.put(w.iloc[indices] if w is not None else None),
                "base_margin": ray.put(b.iloc[indices]
                                       if b is not None else None),
                "label_lower_bound": ray.put(ll.iloc[indices]
                                             if ll is not None else None),
                "label_upper_bound": ray.put(lu.iloc[indices]
                                             if lu is not None else None)
            }
            refs[i] = actor_refs

        return refs, n


class _DistributedRayDMatrixLoader(_RayDMatrixLoader):
    """Load each shard individually."""

    def load_data(self,
                  num_actors: int,
                  sharding: RayShardingMode,
                  rank: Optional[int] = None) -> Tuple[Dict, int]:
        """
        Load data into memory
        """
        if rank is None or not ray.is_initialized:
            raise ValueError(
                "Distributed loading should be done by the actors, not by the"
                "driver program. "
                "\nFIX THIS by refraining from calling `RayDMatrix.load()` "
                "manually for distributed datasets. Hint: You can check if "
                "`RayDMatrix.distributed` is set to True or False.")

        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]

        invalid_data = False
        if isinstance(self.data, str):
            if os.path.isdir(self.data):
                if self.filetype == RayFileType.PARQUET:
                    self.data = sorted(glob.glob(f"{self.data}/**/*.parquet"))
                elif self.filetype == RayFileType.CSV:
                    self.data = sorted(glob.glob(f"{self.data}/**/*.csv"))
                else:
                    invalid_data = True
            elif os.path.exists(self.data):
                self.data = [self.data]
            else:
                print(f"INVALID: {self.data}")
                invalid_data = True

        if not isinstance(self.data, (Iterable, MLDataset)) or invalid_data:
            raise ValueError(
                f"Distributed data loading only works with already "
                f"distributed datasets. These should be specified through a "
                f"list of locations (or single string). "
                f"Got: {type(self.data)}."
                f"\nFIX THIS by passing a list of files (e.g. on S3) to the "
                f"RayDMatrix.")

        if self.label is not None and not isinstance(self.label, str):
            raise ValueError(
                f"Invalid `label` value for distributed datasets: "
                f"{self.label}. Only strings are supported. "
                f"\nFIX THIS by passing a string indicating the label "
                f"column of the dataset as the `label` argument.")

        if isinstance(self.data, MLDataset):
            indices = _get_sharding_indices(sharding, rank, num_actors,
                                            self.data.num_shards())
            local_input_sources = []
        else:
            # Shard input sources
            input_sources = list(self.data)
            n = len(input_sources)

            # Get files this worker should load
            indices = _get_sharding_indices(sharding, rank, num_actors, n)
            local_input_sources = [input_sources[i] for i in indices]

        if isinstance(self.data, MLDataset):
            x, y, w, b, ll, lu = self._load_data_ml_dataset(self.data, indices)
        elif self.filetype == RayFileType.CSV:
            x, y, w, b, ll, lu = self._load_data_csv(local_input_sources)
        elif self.filetype == RayFileType.PARQUET:
            x, y, w, b, ll, lu = self._load_data_parquet(local_input_sources)
        else:
            raise ValueError(
                "Invalid data source type: {} with FileType: {} for a "
                "distributed dataset."
                "\nFIX THIS by passing a supported data type. Supported "
                "data types for distributed datasets are a list of "
                "CSV or Parquet sources.".format(
                    type(self.data), self.filetype))

        if isinstance(x, list):
            n = sum(len(a) for a in x)
        else:
            n = len(x)

        refs = {
            rank: {
                "data": ray.put(x),
                "label": ray.put(y),
                "weight": ray.put(w),
                "base_margin": ray.put(b),
                "label_lower_bound": ray.put(ll),
                "label_upper_bound": ray.put(lu)
            }
        }

        return refs, n


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
            numpy array, Ray MLDataset, modin dataframe, string pointing to
            a csv or parquet file, or list of strings pointing to csv or
            parquet files.
        label: Optional label object. Can be a pandas series, numpy array,
            modin series, string pointing to a csv or parquet file, or
            a string indicating the column of the data dataframe that
            contains the label. If this is not a string it must be of the
            same type as the data argument.
        num_actors: Number of actors to shard this data for. If this is
            not None, data will be loaded and stored into the object store
            after initialization. If this is None, it will be set by
            the ``xgboost_ray.train()`` function, and it will be loaded and
            stored in the object store then. Defaults to None (
        filetype (Optional[RayFileType]): Type of data to read.
            This is disregarded if a data object like a pandas dataframe
            is passed as the ``data`` argument. For filenames,
            the filetype is automaticlly detected via the file name
            (e.g. ``.csv`` will be detected as ``RayFileType.CSV``).
            Passing this argument will overwrite the detected filename.
            If the filename cannot be determined from the ``data`` object,
            passing this is mandatory. Defaults to ``None`` (auto detection).
        ignore (Optional[List[str]]): Exclude these columns from the
            dataframe after loading the data.
        distributed (Optional[bool]): If True, use distributed loading
            (each worker loads a share of the dataset). If False, use
            central loading (the head node loads the whole dataset and
            distributed it). If None, auto-detect and default to
            distributed loading, if possible.
        sharding (RayShardingMode): How to shard the data for different
            workers. ``RayShardingMode.INTERLEAVED`` will divide the data
            per row, i.e. every i-th row will be passed to the first worker,
            every (i+1)th row to the second worker, etc.
            ``RayShardingMode.BATCH`` will divide the data in batches, i.e.
            the first 0-(m-1) rows will be passed to the first worker, the
            m-(2m-1) rows to the second worker, etc. Defaults to
            ``RayShardingMode.INTERLEAVED``. If using distributed data
            loading, sharding happens on a per-file basis, and not on a
            per-row basis, i.e. For interleaved every ith *file* will be
            passed into the first worker, etc.
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
                 missing: Optional[float] = None,
                 weight: Optional[Data] = None,
                 base_margin: Optional[Data] = None,
                 label_lower_bound: Optional[Data] = None,
                 label_upper_bound: Optional[Data] = None,
                 feature_names: Optional[List[str]] = None,
                 feature_types: Optional[List[np.dtype]] = None,
                 num_actors: Optional[int] = None,
                 filetype: Optional[RayFileType] = None,
                 ignore: Optional[List[str]] = None,
                 distributed: Optional[bool] = None,
                 sharding: RayShardingMode = RayShardingMode.INTERLEAVED,
                 lazy: bool = False,
                 **kwargs):

        self._uid = uuid.uuid4().int

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.missing = missing

        self.memory_node_ip = ray.services.get_node_ip_address()
        self.num_actors = num_actors
        self.sharding = sharding

        if distributed is None:
            distributed = _detect_distributed(data)
        else:
            if distributed and not _can_load_distributed(data):
                raise ValueError(
                    f"You passed `distributed=True` to the `RayDMatrix` but "
                    f"the specified data source of type {type(data)} cannot "
                    f"be loaded in a distributed fashion. "
                    f"\nFIX THIS by passing a list of sources (e.g. parquet "
                    f"files stored in a network location) instead.")

        self.distributed = distributed

        if self.distributed:
            self.loader = _DistributedRayDMatrixLoader(
                data=data,
                label=label,
                missing=missing,
                weight=weight,
                base_margin=base_margin,
                label_lower_bound=label_lower_bound,
                label_upper_bound=label_upper_bound,
                feature_names=feature_names,
                feature_types=feature_types,
                filetype=filetype,
                ignore=ignore,
                **kwargs)
        else:
            self.loader = _CentralRayDMatrixLoader(
                data=data,
                label=label,
                missing=missing,
                weight=weight,
                base_margin=base_margin,
                label_lower_bound=label_lower_bound,
                label_upper_bound=label_upper_bound,
                feature_names=feature_names,
                feature_types=feature_types,
                filetype=filetype,
                ignore=ignore,
                **kwargs)

        self.refs: Dict[int, Dict[str, ray.ObjectRef]] = {}
        self.n = None

        self.loaded = False

        if not distributed and num_actors is not None and not lazy:
            self.load_data(num_actors)

    def load_data(self,
                  num_actors: Optional[int] = None,
                  rank: Optional[int] = None):
        if not self.loaded:
            if num_actors is not None:
                if self.num_actors is not None \
                        and num_actors != self.num_actors:
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
            refs, self.n = self.loader.load_data(
                self.num_actors, self.sharding, rank=rank)
            self.refs.update(refs)
            self.loaded = True

    def get_data(
            self, rank: int, num_actors: Optional[int] = None
    ) -> Dict[str, Union[None, pd.DataFrame, List[Optional[pd.DataFrame]]]]:
        self.load_data(num_actors=num_actors, rank=rank)

        refs = self.refs[rank]
        ray.get(list(refs.values()))

        data = {k: ray.get(v) for k, v in refs.items()}

        return data

    def unload_data(self):
        """Delete object references to clear object store"""
        for rank in list(self.refs.keys()):
            for name in list(self.refs[rank].keys()):
                del self.refs[rank][name]
        self.loaded = False

    def __hash__(self):
        return self._uid

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class RayDeviceQuantileDMatrix(RayDMatrix):
    """Currently just a thin wrapper for type detection"""

    def __init__(self, *args, **kwargs):
        if cp is None:
            raise RuntimeError(
                "RayDeviceQuantileDMatrix requires cupy to be installed."
                "\nFIX THIS by installing cupy: `pip install cupy`")
        super(RayDeviceQuantileDMatrix, self).__init__(*args, **kwargs)


def _can_load_distributed(source: Data) -> bool:
    """Returns True if it might be possible to use distributed data loading"""
    if isinstance(source, (int, float, bool)):
        return False
    elif isinstance(source, MLDataset):
        # MLDataset is distributed already
        return True
    elif isinstance(source, str):
        # Strings should point to files or URLs
        return True
    elif isinstance(source, Sequence):
        # Sequence of strings should point to files or URLs
        return isinstance(source[0], str)
    elif isinstance(source, Iterable):
        # If we get an iterable but not a sequence, the best we can do
        # is check if we have a known non-distributed object
        if isinstance(source, (pd.DataFrame, pd.Series, np.ndarray)):
            return False

    # Per default, allow distributed loading.
    return True


def _detect_distributed(source: Data) -> bool:
    """Returns True if we should try to use distributed data loading"""
    if not _can_load_distributed(source):
        return False
    if isinstance(source, MLDataset):
        return True
    if isinstance(source, Iterable) and not isinstance(source, str) and \
       not (isinstance(source, Sequence) and isinstance(source[0], str)):
        # This is an iterable but not a Sequence of strings, and not a
        # pandas dataframe, series, or numpy array.
        # Detect False per default, can be overridden by passing
        # `distributed=True` to the RayDMatrix object.
        return False

    return True


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
        min_len = min(len(d) for d in data)
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
