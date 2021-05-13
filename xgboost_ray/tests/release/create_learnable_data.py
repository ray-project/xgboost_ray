import argparse
import numpy as np
import os
import pandas as pd

from sklearn.datasets import make_classification, make_regression

if __name__ == "__main__":
    if "OMP_NUM_THREADS" in os.environ:
        del os.environ["OMP_NUM_THREADS"]

    parser = argparse.ArgumentParser(description="Create fake data.")
    parser.add_argument("filename", type=str, default="/data/parted.parquet/")
    parser.add_argument(
        "-r",
        "--num-rows",
        required=False,
        type=int,
        default=1e8,
        help="num rows")
    parser.add_argument(
        "-p",
        "--num-partitions",
        required=False,
        type=int,
        default=100,
        help="num partitions")
    parser.add_argument(
        "-c",
        "--num-cols",
        required=False,
        type=int,
        default=4,
        help="num columns (features)")
    parser.add_argument(
        "-C",
        "--num-classes",
        required=False,
        type=int,
        default=2,
        help="num classes")
    parser.add_argument(
        "-s",
        "--seed",
        required=False,
        type=int,
        default=1234,
        help="random seed")
    parser.add_argument(
        "-T",
        "--target",
        required=False,
        type=float,
        default=0.8,
        help="target accuracy")

    args = parser.parse_args()

    seed = int(args.seed)
    np.random.seed(seed)

    num_rows = int(args.num_rows)
    num_cols = int(args.num_cols)
    num_classes = int(args.num_classes)
    target = float(args.target)

    if num_classes > 0:
        x, y = make_classification(
            n_samples=num_rows,
            n_features=num_cols,
            n_informative=num_cols // 2,
            n_redundant=num_cols // 10,
            n_repeated=0,
            n_classes=num_classes,
            n_clusters_per_class=2,
            flip_y=1 - target,
            random_state=seed,
        )
    else:
        x, y = make_regression(
            n_samples=num_rows,
            n_features=num_cols,
            n_informative=num_cols // 2,
            n_targets=1,
            noise=0.1,
            random_state=seed,
        )

    filename = args.filename
    num_partitions = args.num_partitions

    data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(num_cols)])

    rows_per_partition = np.floor(len(data) / num_partitions)

    partition_arr = np.repeat(
        np.arange(num_partitions), repeats=rows_per_partition)
    if len(partition_arr) < len(data):
        # If this was not evenly divided, append
        missing = len(data) - len(partition_arr)
        partition_arr = np.append(partition_arr, np.arange(missing))

    partition = pd.Series(partition_arr, copy=False, dtype=np.int32)

    data["labels"] = y
    data["partition"] = partition

    os.makedirs(filename, 0o755, exist_ok=True)

    # Write partition-wise to avoid OOM errors
    for i in range(num_partitions):
        part = data[partition_arr == i]
        part.to_parquet(
            filename,
            partition_cols=["partition"],
            engine="pyarrow",
            partition_filename_cb=lambda key: f"part_{key[0]}.parquet")
