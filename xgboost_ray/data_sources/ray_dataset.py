from typing import Any, Optional, Sequence, Dict, Union, Tuple

import pandas as pd

import ray
from ray.actor import ActorHandle
from ray.data.dataset import Dataset

from xgboost_ray.data_sources.data_source import DataSource, RayFileType
from xgboost_ray.data_sources.pandas import Pandas

try:
    import ray.data.dataset  # noqa: F401
    RAY_DATASET_AVAILABLE = True
except (ImportError, AttributeError):
    RAY_DATASET_AVAILABLE = False


def _assert_ray_data_available():
    if not RAY_DATASET_AVAILABLE:
        raise RuntimeError(
            "Tried to use Ray datasets as a data source, but your version "
            "of Ray does not support it. "
            "\nFIX THIS by upgrading Ray: `pip install -U ray`. "
            "\nPlease also raise an issue on our GitHub: "
            "https://github.com/ray-project/xgboost_ray as this part of "
            "the code should not have been reached.")


class RayDataset(DataSource):
    """Read from distributed Ray dataset."""
    supports_central_loading = True
    supports_distributed_loading = True

    @staticmethod
    def is_data_type(data: Any,
                     filetype: Optional[RayFileType] = None) -> bool:
        if not RAY_DATASET_AVAILABLE:
            return False

        return isinstance(data, ray.data.dataset.Dataset)

    @staticmethod
    def load_data(
            data: Any,  # ray.data.dataset.Dataset
            ignore: Optional[Sequence[str]] = None,
            indices: Optional[Union[Sequence[int], Sequence[
                Dataset]]] = None,
            **kwargs) -> pd.DataFrame:
        _assert_ray_data_available()

        if indices is not None and len(indices) > 0 and isinstance(
                indices[0], Dataset):
            # We got a list of ObjectRefs belonging to Ray dataset partition
            data = indices
            indices = None

        if indices is not None:
            data = [data[i] for i in indices]
        local_df = [ds.to_pandas(limit=float("inf")) for ds in data]
        return Pandas.load_data(pd.concat(local_df, copy=False), ignore=ignore)

    @staticmethod
    def convert_to_series(data: Any) -> pd.Series:
        _assert_ray_data_available()

        if isinstance(data, Dataset):
            data = data.to_pandas(limit=float("inf"))
        else:
            data = pd.concat([ds.to_pandas(limit=float("inf")) for ds in data], copy=False)
        return DataSource.convert_to_series(data)

    @staticmethod
    def get_actor_shards(
            data: Any,  # ray.data.dataset.Dataset
            actors: Sequence[ActorHandle]) -> \
            Tuple[Any, Optional[Dict[int, Any]]]:
        _assert_ray_data_available()

        dataset_splits = data.split(
            len(actors),
            equal=True,
            locality_hints=actors,
        )

        # Ray datasets should not be serialized
        return None, {i: [dataset_split] for i, dataset_split in enumerate(dataset_splits)}

    @staticmethod
    def get_n(data: Any):
        """
        Return number of distributed blocks.
        """
        return data.num_blocks()
