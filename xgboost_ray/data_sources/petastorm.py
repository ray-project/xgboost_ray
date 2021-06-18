from typing import Any, Optional, Sequence, Union, List

import pandas as pd

from xgboost_ray.data_sources.data_source import DataSource, RayFileType

try:
    import petastorm
    PETASTORM_INSTALLED = True
except ImportError:
    PETASTORM_INSTALLED = False


def _assert_petastorm_installed():
    if not PETASTORM_INSTALLED:
        raise RuntimeError(
            "Tried to use Petastorm as a data source, but petastorm is not "
            "installed. This function shouldn't have been called. "
            "\nFIX THIS by installing petastorm: `pip install petastorm`. "
            "\nPlease also raise an issue on our GitHub: "
            "https://github.com/ray-project/xgboost_ray as this part of "
            "the code should not have been reached.")


class Petastorm(DataSource):
    """Read with Petastorm.

    `Petastorm <https://github.com/uber/petastorm>`_ is a machine learning
    training and evaluation library.

    This class accesses Petastorm's dataset loading interface for efficient
    loading of large datasets.
    """
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        return PETASTORM_INSTALLED and filetype == RayFileType.PETASTORM

    @staticmethod
    def get_filetype(data: Any) -> Optional[RayFileType]:
        if not PETASTORM_INSTALLED:
            return None

        if not isinstance(data, List):
            data = [data]

        def _is_compatible(url: str):
            return url.endswith(".parquet") and (url.startswith("s3://")
                                                 or url.startswith("gs://")
                                                 or url.startswith("hdfs://")
                                                 or url.startswith("file://"))

        if all(_is_compatible(url) for url in data):
            return RayFileType.PETASTORM

        return None

    @staticmethod
    def load_data(data: Union[str, Sequence[str]],
                  ignore: Optional[Sequence[str]] = None,
                  indices: Optional[Sequence[int]] = None,
                  **kwargs) -> pd.DataFrame:
        _assert_petastorm_installed()
        with petastorm.make_batch_reader(data) as reader:
            shards = [
                pd.DataFrame(batch._asdict()) for i, batch in enumerate(reader)
                if not indices or i in indices
            ]

        local_df = pd.concat(shards, copy=False)

        if ignore:
            local_df = local_df[local_df.columns.difference(ignore)]

        return local_df

    @staticmethod
    def get_n(data: Any):
        return len(list(data))
