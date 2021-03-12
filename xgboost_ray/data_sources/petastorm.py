from typing import Any, Optional, Sequence, Union, List

import pandas as pd

from xgboost_ray.data_sources.data_source import DataSource, RayFileType

try:
    import petastorm
    PETASTORM_INSTALLED = True
except ImportError:
    PETASTORM_INSTALLED = False


class Petastorm(DataSource):
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
        with petastorm.make_batch_reader(data) as reader:
            shards = [pd.DataFrame(batch._asdict()) for batch in reader]

        local_df = pd.concat(shards, copy=False)

        if ignore:
            local_df = local_df[local_df.columns.difference(ignore)]

        return local_df
