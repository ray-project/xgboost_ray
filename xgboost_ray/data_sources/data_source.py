from enum import Enum
from typing import Any, Optional, Sequence, Tuple

import pandas as pd


class RayFileType(Enum):
    """Enum for different file types (used for overrides)."""
    CSV = 1
    PARQUET = 2
    PETASTORM = 3


class DataSource:
    supports_central_loading = True
    supports_distributed_loading = False

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        """Check if the supplied data matches this data source.

        Args:
            data (Any): Data set
            filetype (Optional[RayFileType]): Optional RayFileType. Some
                data sources might require that this is explicitly set
                (e.g. if multiple sources can read CSV files).

        Returns:
            Boolean indicating if this data source belongs to/is compatible
                with the data.
        """
        return False

    @staticmethod
    def get_filetype(data: Any) -> Optional[RayFileType]:
        """Method to help infer the filetype.

        Should return None if the supplied data type (usually a filename)
        is not covered by this data source, otherwise the filetype should
        be returned.

        Args:
            data (Any): Data set

        Returns:
            RayFileType or None.
        """
        return None

    @staticmethod
    def load_data(data: Any,
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load data into a pandas dataframe.

        Ignore specific columns, and optionally select specific indices.

        Args:
            data (Any): Input data
            ignore (Optional[Sequence[str]]): Column names to ignore
            indices (Optional[Sequence[int]]): Indices to select. What an
                index indicates depends on the data source.

        Returns:
            Pandas DataFrame.
        """
        raise NotImplementedError

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
        return len(list(data))