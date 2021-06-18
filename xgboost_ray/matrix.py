import glob
import math
import uuid
from enum import Enum
from typing import Union, Optional, Tuple, Iterable, List, Dict, Sequence, \
    Callable, Type

from ray.actor import ActorHandle

try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np
import pandas as pd
import xgboost as xgb

import os

import ray
from ray import logger

from xgboost_ray.util import Unavailable
from xgboost_ray.data_sources import DataSource, data_sources, RayFileType

try:
    from ray.util.data import MLDataset
except ImportError:
    MLDataset = Unavailable

try:
    from xgboost.core import DataIter
    LEGACY_MATRIX = False
except ImportError:
    DataIter = object
    LEGACY_MATRIX = True

Data = Union[str, List[str], np.ndarray, pd.DataFrame, pd.Series, MLDataset]


def concat_dataframes(dfs: List[Optional[pd.DataFrame]]):
    filtered = [df for df in dfs if df is not None]
    return pd.concat(filtered, ignore_index=True, copy=False)


class RayShardingMode(Enum):
    """Enum for different modes of sharding the data.

    ``RayShardingMode.INTERLEAVED`` will divide the data
    per row, i.e. every i-th row will be passed to the first worker,
    every (i+1)th row to the second worker, etc.

    ``RayShardingMode.BATCH`` will divide the data in batches, i.e.
    the first 0-(m-1) rows will be passed to the first worker, the
    m-(2m-1) rows to the second worker, etc.

    ``RayShardingMode.FIXED`` is set automatically when using a distributed
    data source that assigns actors to specific data shards on initialization
    and then keeps these fixed.
    """
    INTERLEAVED = 1
    BATCH = 2
    FIXED = 3


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

        self.data_source = None
        self.actor_shards = None

        self.filetype = filetype
        self.ignore = ignore
        self.kwargs = kwargs

        self._cached_n = None

        check = None
        if isinstance(data, str):
            check = data
        elif isinstance(data, Sequence) and isinstance(data[0], str):
            check = data[0]

        if check is not None:
            if not self.filetype:
                # Try to guess filetype
                for data_source in data_sources:
                    self.filetype = data_source.get_filetype(check)
                    if self.filetype:
                        break
                if not self.filetype:
                    raise ValueError(
                        f"File or stream ({check}) specified as data source, "
                        "but filetype could not be detected. "
                        "\nFIX THIS by passing "
                        "the `filetype` parameter to the RayDMatrix. Use the "
                        "`RayFileType` enum for this.")

    def get_data_source(self) -> Type[DataSource]:
        raise NotImplementedError

    def assert_enough_shards_for_actors(self, num_actors: int):
        """Assert that we have enough shards to split across actors."""
        # Pass per default
        pass

    def update_matrix_properties(self, matrix: xgb.DMatrix):
        data_source = self.get_data_source()
        data_source.update_feature_names(matrix, self.feature_names)

    def assign_shards_to_actors(self, actors: Sequence[ActorHandle]) -> bool:
        """Assign data shards to actors.

        Returns True if shards were assigned to actors. In that case, the
        sharding mode should be adjusted to ``RayShardingMode.FIXED``.
        Returns False otherwise.
        """
        return False

    def _split_dataframe(
            self, local_data: pd.DataFrame, data_source: Type[DataSource]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series],
               Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Split dataframe into

        `features`, `labels`, `weight`, `base_margin`, `label_lower_bound`,
        `label_upper_bound`

        """
        exclude_cols: List[str] = []  # Exclude these columns from `x`

        label, exclude = data_source.get_column(local_data, self.label)
        if exclude:
            exclude_cols.append(exclude)

        weight, exclude = data_source.get_column(local_data, self.weight)
        if exclude:
            exclude_cols.append(exclude)

        base_margin, exclude = data_source.get_column(local_data,
                                                      self.base_margin)
        if exclude:
            exclude_cols.append(exclude)

        label_lower_bound, exclude = data_source.get_column(
            local_data, self.label_lower_bound)
        if exclude:
            exclude_cols.append(exclude)

        label_upper_bound, exclude = data_source.get_column(
            local_data, self.label_upper_bound)
        if exclude:
            exclude_cols.append(exclude)

        x = local_data
        if exclude_cols:
            x = x[x.columns.difference(exclude_cols)]

        return x, label, weight, base_margin, label_lower_bound, \
            label_upper_bound

    def load_data(self,
                  num_actors: int,
                  sharding: RayShardingMode,
                  rank: Optional[int] = None) -> Tuple[Dict, int]:
        raise NotImplementedError


class _CentralRayDMatrixLoader(_RayDMatrixLoader):
    """Load full dataset from a central location and put into object store"""

    def get_data_source(self) -> Type[DataSource]:
        if self.data_source:
            return self.data_source

        data_source = None
        for source in data_sources:
            if not source.supports_central_loading:
                continue

            try:
                if source.is_data_type(self.data, self.filetype):
                    data_source = source
                    break
            except Exception as exc:
                # If checking the data throws an exception, the data source
                # is not available.
                logger.warning(
                    f"Checking data source {source.__name__} failed "
                    f"with exception: {exc}")
                continue

        if not data_source:
            raise ValueError(
                "Unknown data source type: {} with FileType: {}."
                "\nFIX THIS by passing a supported data type. Supported "
                "data types include pandas.DataFrame, pandas.Series, "
                "np.ndarray, and CSV/Parquet file paths. If you specify a "
                "file, path, consider passing the `filetype` argument to "
                "specify the type of the source. Use the `RayFileType` "
                "enum for that. If using Modin, Dask, or Petastorm, "
                "make sure the library is installed.".format(
                    type(self.data), self.filetype))

        if self.label is not None and not isinstance(self.label, str) and \
                not type(self.data) != type(self.label):  # noqa: E721:
            # Label is an object of a different type than the main data.
            # We have to make sure they are compatible
            if not data_source.is_data_type(self.label):
                raise ValueError(
                    "The passed `data` and `label` types are not compatible."
                    "\nFIX THIS by passing the same types to the "
                    "`RayDMatrix` - e.g. a `pandas.DataFrame` as `data` "
                    "and `label`. The `label` can always be a string. Got "
                    "{} for the main data and {} for the label.".format(
                        type(self.data), type(self.label)))

        self.data_source = data_source
        self._cached_n = data_source.get_n(self.data)
        return self.data_source

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

        data_source = self.get_data_source()

        max_num_shards = self._cached_n or data_source.get_n(self.data)
        if num_actors > max_num_shards:
            raise RuntimeError(
                f"Trying to shard data for {num_actors} actors, but the "
                f"maximum number of shards (i.e. the number of data rows) "
                f"is {max_num_shards}. Consider using fewer actors.")

        # We're doing central data loading here, so we don't pass any indices,
        # yet. Instead, we'll be selecting the rows below.
        local_df = data_source.load_data(
            self.data, ignore=self.ignore, indices=None, **self.kwargs)
        x, y, w, b, ll, lu = self._split_dataframe(
            local_df, data_source=data_source)

        if isinstance(x, list):
            n = sum(len(a) for a in x)
        else:
            n = len(x)

        refs = {}
        for i in range(num_actors):
            # Here we actually want to split the data.
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

    def get_data_source(self) -> Type[DataSource]:
        if self.data_source:
            return self.data_source

        invalid_data = False
        if isinstance(self.data, str):
            if self.filetype == RayFileType.PETASTORM:
                self.data = [self.data]
            elif os.path.isdir(self.data):
                if self.filetype == RayFileType.PARQUET:
                    self.data = sorted(glob.glob(f"{self.data}/**/*.parquet"))
                elif self.filetype == RayFileType.CSV:
                    self.data = sorted(glob.glob(f"{self.data}/**/*.csv"))
                else:
                    invalid_data = True
            elif os.path.exists(self.data):
                self.data = [self.data]
            else:
                invalid_data = True

        # Todo (krfricke): It would be good to have a more general way to
        # check for compatibility here. Combine with test below?
        if not isinstance(self.data, (Iterable, MLDataset)) or invalid_data:
            raise ValueError(
                f"Distributed data loading only works with already "
                f"distributed datasets. These should be specified through a "
                f"list of locations (or a single string). "
                f"Got: {type(self.data)}."
                f"\nFIX THIS by passing a list of files (e.g. on S3) to the "
                f"RayDMatrix.")

        if self.label is not None and not isinstance(self.label, str):
            raise ValueError(
                f"Invalid `label` value for distributed datasets: "
                f"{self.label}. Only strings are supported. "
                f"\nFIX THIS by passing a string indicating the label "
                f"column of the dataset as the `label` argument.")

        data_source = None
        for source in data_sources:
            if not source.supports_distributed_loading:
                continue

            try:
                if source.is_data_type(self.data, self.filetype):
                    data_source = source
                    break
            except Exception as exc:
                # If checking the data throws an exception, the data source
                # is not available.
                logger.warning(
                    f"Checking data source {source.__name__} failed "
                    f"with exception: {exc}")
                continue

        if not data_source:
            raise ValueError(
                f"Invalid data source type: {type(self.data)} "
                f"with FileType: {self.filetype} for a distributed dataset."
                "\nFIX THIS by passing a supported data type. Supported "
                "data types for distributed datasets are a list of "
                "CSV or Parquet sources as well as Ray MLDatasets. If using "
                "Modin, Dask, or Petastorm, make sure the library is "
                "installed.")

        self.data_source = data_source
        self._cached_n = data_source.get_n(self.data)
        return self.data_source

    def assert_enough_shards_for_actors(self, num_actors: int):
        data_source = self.get_data_source()

        max_num_shards = self._cached_n or data_source.get_n(self.data)
        if num_actors > max_num_shards:
            raise RuntimeError(
                f"Trying to shard data for {num_actors} actors, but the "
                f"maximum number of shards is {max_num_shards}. If you "
                f"want to shard the dataset by rows, consider "
                f"centralized loading by passing `distributed=False` to "
                f"the `RayDMatrix`. Otherwise consider using fewer actors "
                f"or re-partitioning your data.")

    def assign_shards_to_actors(self, actors: Sequence[ActorHandle]) -> bool:
        if not isinstance(self.label, str):
            # Currently we only support fixed data sharding for datasets
            # that contain both the label and the data.
            return False

        if self.actor_shards:
            # Only assign once
            return True

        data_source = self.get_data_source()
        data, actor_shards = data_source.get_actor_shards(self.data, actors)
        if not actor_shards:
            return False

        self.data = data
        self.actor_shards = actor_shards
        return True

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

        data_source = self.get_data_source()

        if self.actor_shards:
            if rank is None:
                raise RuntimeError(
                    "Distributed loading requires a rank to be passed, "
                    "got None")
            rank_shards = self.actor_shards[rank]
            local_df = data_source.load_data(
                self.data,
                indices=rank_shards,
                ignore=self.ignore,
                **self.kwargs)
            x, y, w, b, ll, lu = self._split_dataframe(
                local_df, data_source=data_source)

            if isinstance(x, list):
                n = sum(len(a) for a in x)
            else:
                n = len(x)
        else:
            n = self._cached_n or data_source.get_n(self.data)
            indices = _get_sharding_indices(sharding, rank, num_actors, n)

            if not indices:
                x, y, w, b, ll, lu = None, None, None, None, None, None
                n = 0
            else:
                local_df = data_source.load_data(
                    self.data,
                    ignore=self.ignore,
                    indices=indices,
                    **self.kwargs)
                x, y, w, b, ll, lu = self._split_dataframe(
                    local_df, data_source=data_source)

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
                 weight: Optional[Data] = None,
                 base_margin: Optional[Data] = None,
                 missing: Optional[float] = None,
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

    @property
    def has_label(self):
        return self.loader.label is not None

    def assign_shards_to_actors(self, actors: Sequence[ActorHandle]) -> bool:
        success = self.loader.assign_shards_to_actors(actors)
        if success:
            self.sharding = RayShardingMode.FIXED
        return success

    def assert_enough_shards_for_actors(self, num_actors: int):
        self.loader.assert_enough_shards_for_actors(num_actors=num_actors)

    def load_data(self,
                  num_actors: Optional[int] = None,
                  rank: Optional[int] = None):
        """Load data, putting it into the Ray object store.

        If a rank is given, only data for this rank is loaded (for
        distributed data sources only).
        """
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
        """Get data, i.e. return dataframe for a specific actor.

        This method is called from an actor, given its rank and the
        total number of actors. If the data is not yet loaded, loading
        is triggered.
        """
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

    def update_matrix_properties(self, matrix: xgb.DMatrix):
        self.loader.update_matrix_properties(matrix)

    def __hash__(self):
        return self._uid

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class RayDeviceQuantileDMatrix(RayDMatrix):
    """Currently just a thin wrapper for type detection"""

    def __init__(self,
                 data: Data,
                 label: Optional[Data] = None,
                 weight: Optional[Data] = None,
                 base_margin: Optional[Data] = None,
                 missing: Optional[float] = None,
                 label_lower_bound: Optional[Data] = None,
                 label_upper_bound: Optional[Data] = None,
                 feature_names: Optional[List[str]] = None,
                 feature_types: Optional[List[np.dtype]] = None,
                 *args,
                 **kwargs):
        if cp is None:
            raise RuntimeError(
                "RayDeviceQuantileDMatrix requires cupy to be installed."
                "\nFIX THIS by installing cupy: `pip install cupy-cudaXYZ` "
                "where XYZ is your local CUDA version.")
        if label_lower_bound or label_upper_bound:
            raise RuntimeError(
                "RayDeviceQuantileDMatrix does not support "
                "`label_lower_bound` and `label_upper_bound` (just as the "
                "xgboost.DeviceQuantileDMatrix). Please pass None instead.")
        super(RayDeviceQuantileDMatrix, self).__init__(
            data=data,
            label=label,
            weight=weight,
            base_margin=base_margin,
            missing=missing,
            label_lower_bound=None,
            label_upper_bound=None,
            feature_names=feature_names,
            feature_types=feature_types,
            *args,
            **kwargs)

    def get_data(
            self, rank: int, num_actors: Optional[int] = None
    ) -> Dict[str, Union[None, pd.DataFrame, List[Optional[pd.DataFrame]]]]:
        data_dict = super(RayDeviceQuantileDMatrix, self).get_data(
            rank=rank, num_actors=num_actors)
        # Remove some dict keys here that are generated automatically
        data_dict.pop("label_lower_bound", None)
        data_dict.pop("label_upper_bound", None)
        return data_dict


def _can_load_distributed(source: Data) -> bool:
    """Returns True if it might be possible to use distributed data loading"""
    from xgboost_ray.data_sources.ml_dataset import MLDataset
    from xgboost_ray.data_sources.modin import Modin

    if isinstance(source, (int, float, bool)):
        return False
    elif MLDataset.is_data_type(source):
        return True
    elif Modin.is_data_type(source):
        return True
    elif isinstance(source, str):
        # Strings should point to files or URLs
        # Usually parquet files point to directories
        return source.endswith(".parquet")
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
    from xgboost_ray.data_sources.ml_dataset import MLDataset
    from xgboost_ray.data_sources.modin import Modin
    if not _can_load_distributed(source):
        return False
    if MLDataset.is_data_type(source):
        return True
    if Modin.is_data_type(source):
        return True
    if isinstance(source, Iterable) and not isinstance(source, str) and \
       not (isinstance(source, Sequence) and isinstance(source[0], str)):
        # This is an iterable but not a Sequence of strings, and not a
        # pandas dataframe, series, or numpy array.
        # Detect False per default, can be overridden by passing
        # `distributed=True` to the RayDMatrix object.
        return False

    # Otherwise, assume distributed loading is possible
    return True


def _get_sharding_indices(sharding: RayShardingMode, rank: int,
                          num_actors: int, n: int):
    """Return indices that belong to worker with rank `rank`"""
    if sharding == RayShardingMode.BATCH:
        start_index = int(rank * math.ceil(n / num_actors))
        end_index = int((rank + 1) * math.ceil(n / num_actors))
        end_index = min(end_index, n)
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
    if sharding not in (RayShardingMode.BATCH, RayShardingMode.INTERLEAVED):
        raise ValueError(f"Invalid value for `sharding` parameter: "
                         f"{sharding}"
                         f"\nFIX THIS by passing any item of the "
                         f"`RayShardingMode` enum, for instance "
                         f"`RayShardingMode.BATCH`.")

    # discard empty arrays that show up with BATCH
    data = [d for d in data if len(d)]

    if data[0].ndim == 1:
        # most common case
        if sharding == RayShardingMode.BATCH:
            res = np.concatenate(data)
        elif sharding == RayShardingMode.INTERLEAVED:
            # Sometimes the lengths are off by 1 for uneven divisions
            min_len = min(len(d) for d in data)
            res = np.ravel(np.column_stack([d[0:min_len] for d in data]))
            # Append these here
            res = np.concatenate(
                [res] + [d[min_len:] for d in data if len(d) > min_len])
    else:
        # objective="multi:softprob" returns n-dimensional arrays that
        # need to be handled differently
        if sharding == RayShardingMode.BATCH:
            res = np.vstack(data)
        elif sharding == RayShardingMode.INTERLEAVED:
            # Sometimes the lengths are off by 1 for uneven divisions
            min_len = min(len(d) for d in data)
            # the number of classes will be constant, though
            class_len = data[0].shape[1]
            min_len_data = [d[0:min_len] for d in data]
            res = np.hstack(min_len_data).reshape(
                len(min_len_data) * min_len, class_len)
            # Append these here
            res = np.concatenate(
                [res] + [d[min_len:] for d in data if len(d) > min_len])
    return res
