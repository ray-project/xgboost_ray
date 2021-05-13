from typing import Any, Optional, Sequence, Tuple, Dict, List

from enum import Enum

import pandas as pd
import xgboost as xgb

from ray.actor import ActorHandle


class RayFileType(Enum):
    """Enum for different file types (used for overrides)."""
    CSV = 1
    PARQUET = 2
    PETASTORM = 3


class DataSource:
    """Abstract class for data sources.

    xgboost_ray supports reading from various sources, such as files
    (e.g. CSV, Parquet) or distributed datasets (Ray MLDataset, Modin).

    This abstract class defines an interface to read from these sources.
    New data sources can be added by implementing this interface.

    ``DataSource`` classes are not instantiated. Instead, static and
    class methods are called directly.
    """
    supports_central_loading = True
    supports_distributed_loading = False

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        """Check if the supplied data matches this data source.

        Args:
            data (Any): Dataset.
            filetype (Optional[RayFileType]): RayFileType of the provided
                dataset. Some DataSource implementations might require
                that this is explicitly set (e.g. if multiple sources can
                read CSV files).

        Returns:
            Boolean indicating if this data source belongs to/is compatible
                with the data.
        """
        return False

    @staticmethod
    def get_filetype(data: Any) -> Optional[RayFileType]:
        """Method to help infer the filetype.

        Returns None if the supplied data type (usually a filename)
        is not covered by this data source, otherwise the filetype
        is returned.

        Args:
            data (Any): Data set

        Returns:
            RayFileType or None.
        """
        return None

    @staticmethod
    def load_data(data: Any,
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[Any]] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load data into a pandas dataframe.

        Ignore specific columns, and optionally select specific indices.

        Args:
            data (Any): Input data
            ignore (Optional[Sequence[str]]): Column names to ignore
            indices (Optional[Sequence[Any]]): Indices to select. What an
                index indicates depends on the data source.

        Returns:
            Pandas DataFrame.
        """
        raise NotImplementedError

    @staticmethod
    def update_feature_names(matrix: xgb.DMatrix,
                             feature_names: Optional[List[str]]):
        """Optionally update feature names before training/prediction

        Args:
            matrix (xgb.DMatrix): xgboost DMatrix object.
            feature_names (List[str]): Feature names manually passed to the
                ``RayDMatrix`` object.

        """
        pass

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        """Convert data from the data source type to a pandas series"""
        if isinstance(data, pd.DataFrame):
            return pd.Series(data.squeeze())

        if not isinstance(data, pd.Series):
            return pd.Series(data)

        return data

    @classmethod
    def get_column(cls, data: pd.DataFrame,
                   column: Any) -> Tuple[pd.Series, Optional[str]]:
        """Helper method wrapping around convert to series.

        This method should usually not be overwritten.
        """
        if isinstance(column, str):
            return data[column], column
        elif column is not None:
            return cls.convert_to_series(column), None
        return column, None

    @staticmethod
    def get_n(data: Any):
        """Get length of data source partitions for sharding."""
        return len(data)

    @staticmethod
    def get_actor_shards(
            data: Any,
            actors: Sequence[ActorHandle]) -> \
            Tuple[Any, Optional[Dict[int, Any]]]:
        """Get a dict mapping actor ranks to shards.

        Args:
            data (Any): Data to shard.

        Returns:
            Returns a tuple of which the first element indicates the new
                data object that will overwrite the existing data object
                in the RayDMatrix (e.g. when the object is not serializable).
                The second element is a dict mapping actor ranks to shards.
                These objects are usually passed to the ``load_data()`` method
                for distributed loading, so that method needs to be able to
                deal with the respective data.
        """
        return data, None
