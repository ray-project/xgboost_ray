import os
import time

from xgboost_ray import train, RayDMatrix

import pandas as pd


def main():
    # Example adapted from this blog post:
    # https://medium.com/rapids-ai/a-new-official-dask-api-for-xgboost-e8b10f3d1eb7
    # This uses the HIGGS dataset. Download here:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
    fname = "HIGGS.csv"
    colnames = ["label"] + ["feature-%02d" % i for i in range(1, 29)]

    dtrain = RayDMatrix(
        os.path.abspath(fname), label="label", names=colnames)

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

    bst.save_model('higgs.xgb')
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))


if __name__ == "__main__":
    import ray
    ray.init()

    start = time.time()
    main()
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
