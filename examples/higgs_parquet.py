import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .higgs import download_higgs
from xgboost_ray import train, RayDMatrix

FILENAME_CSV = "HIGGS.csv.gz"
FILENAME_PARQUET = "HIGGS.parquet"


def csv_to_parquet(in_file, out_file, chunksize=100_000, **csv_kwargs):
    if os.path.exists(out_file):
        return False

    print(f"Converting CSV {in_file} to PARQUET {out_file}")
    csv_stream = pd.read_csv(
        in_file, sep=",", chunksize=chunksize, low_memory=False, **csv_kwargs)

    parquet_schema = None
    parquet_writer = None
    for i, chunk in enumerate(csv_stream):
        print("Chunk", i)
        if not parquet_schema:
            # Guess the schema of the CSV file from the first chunk
            parquet_schema = pa.Table.from_pandas(df=chunk).schema
            # Open a Parquet file for writing
            parquet_writer = pq.ParquetWriter(
                out_file, parquet_schema, compression="snappy")
        # Write CSV chunk to the parquet file
        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)

    parquet_writer.close()
    return True


def main():
    # Example adapted from this blog post:
    # https://medium.com/rapids-ai/a-new-official-dask-api-for-xgboost-e8b10f3d1eb7
    # This uses the HIGGS dataset. Download here:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz

    if not os.path.exists(FILENAME_PARQUET):
        if not os.path.exists(FILENAME_CSV):
            download_higgs(FILENAME_CSV)
            print("Downloaded HIGGS csv dataset")
        print("Converting HIGGS csv dataset to parquet")
        csv_to_parquet(
            FILENAME_CSV,
            FILENAME_PARQUET,
            names=[
                "label", "feature-01", "feature-02", "feature-03",
                "feature-04", "feature-05", "feature-06", "feature-07",
                "feature-08", "feature-09", "feature-10", "feature-11",
                "feature-12", "feature-13", "feature-14", "feature-15",
                "feature-16", "feature-17", "feature-18", "feature-19",
                "feature-20", "feature-21", "feature-22", "feature-23",
                "feature-24", "feature-25", "feature-26", "feature-27",
                "feature-28"
            ])

    colnames = ["label"] + ["feature-%02d" % i for i in range(1, 29)]

    # Here we load the Parquet file
    dtrain = RayDMatrix(
        os.path.abspath(FILENAME_PARQUET), label="label", columns=colnames)

    config = {
        "tree_method": "hist",
        "eval_metric": ["logloss", "error"],
    }

    evals_result = {}

    start = time.time()
    bst = train(
        config,
        dtrain,
        evals_result=evals_result,
        max_actor_restarts=1,
        num_boost_round=100,
        evals=[(dtrain, "train")])
    taken = time.time() - start
    print(f"TRAIN TIME TAKEN: {taken:.2f} seconds")

    bst.save_model("higgs.xgb")
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))


if __name__ == "__main__":
    import ray
    ray.init()

    start = time.time()
    main()
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
