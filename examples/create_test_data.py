import os

import pandas as pd
import numpy as np


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


def main():
    create_parquet(
        "parted.parquet",
        num_rows=1_000_000_000,
        num_partitions=1_000,
        num_features=8,
        num_classes=2)


if __name__ == "__main__":
    main()
