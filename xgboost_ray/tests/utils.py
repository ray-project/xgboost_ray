import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd


def create_data(num_rows: int, num_cols: int, dtype: np.dtype = np.float32):

    return pd.DataFrame(
        np.random.uniform(0.0, 10.0, size=(num_rows, num_cols)),
        columns=[f"feature_{i}" for i in range(num_cols)],
        dtype=dtype)


def create_labels(num_rows: int,
                  num_classes: int = 2,
                  dtype: np.dtype = np.int32):

    return pd.Series(
        np.random.randint(0, num_classes, size=num_rows),
        dtype=dtype,
        name="label")


def create_parquet(filename: str,
                   num_rows: int,
                   num_features: int,
                   num_classes: int = 2,
                   num_partitions: int = 1):

    partition_rows = num_rows // num_partitions
    for partition in range(num_partitions):
        print(f"Creating partition {partition}")
        data = create_data(partition_rows, num_features)
        labels = create_labels(partition_rows, num_classes)
        partition = pd.Series(
            np.full(partition_rows, partition), dtype=np.int32)

        data["labels"] = labels
        data["partition"] = partition

        os.makedirs(filename, 0o755, exist_ok=True)
        data.to_parquet(filename, partition_cols=["partition"])


def create_parquet_in_tempdir(filename: str,
                              num_rows: int,
                              num_features: int,
                              num_classes: int = 2,
                              num_partitions: int = 1) -> Tuple[str, str]:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, filename)
    create_parquet(
        path,
        num_rows=num_rows,
        num_features=num_features,
        num_classes=num_classes,
        num_partitions=num_partitions)
    return temp_dir, path
