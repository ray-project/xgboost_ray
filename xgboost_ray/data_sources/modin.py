from typing import Any, Optional, Sequence

from xgboost_ray.data_sources.data_source import DataSource, RayFileType

import pandas as pd

try:
    import modin  # noqa: F401
    MODIN_INSTALLED = True
except ImportError:
    MODIN_INSTALLED = False


def _assert_modin_installed():
    if not MODIN_INSTALLED:
        raise RuntimeError(
            "Tried to use Modin as a data source, but modin is not "
            "installed. This function shouldn't have been called. "
            "\nFIX THIS by installing modin: `pip install modin`. "
            "\nPlease also raise an issue on our GitHub: "
            "https://github.com/ray-project/xgboost_ray as this part of "
            "the code should not have been reached.")


class Modin(DataSource):
    """Read from distributed Modin dataframe.

    `Modin <https://github.com/modin-project/modin>`_ is a distributed
    drop-in replacement for pandas supporting Ray as a backend.

    Modin dataframes are stored on multiple actors, making them
    suitable for distributed loading.
    """

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        if not MODIN_INSTALLED:
            return False
        from modin.pandas import DataFrame as ModinDataFrame, \
            Series as ModinSeries

        return isinstance(data, (ModinDataFrame, ModinSeries))

    @staticmethod
    def load_data(
            data: Any,  # modin.pandas.DataFrame
            ignore: Optional[Sequence[str]] = None,
            indices: Optional[Sequence[int]] = None,
            **kwargs) -> pd.DataFrame:
        _assert_modin_installed()
        local_df = data
        if indices:
            local_df = local_df.iloc(indices)

        local_df = local_df._to_pandas()

        if ignore:
            local_df = local_df[local_df.columns.difference(ignore)]

        return local_df

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        _assert_modin_installed()
        from modin.pandas import DataFrame as ModinDataFrame, \
            Series as ModinSeries

        if isinstance(data, ModinDataFrame):
            return pd.Series(data._to_pandas().squeeze())
        elif isinstance(data, ModinSeries):
            return data._to_pandas()

        return DataSource.convert_to_series(data)
