import argparse
import os
import shutil
import time

from xgboost_ray import train, RayDMatrix, RayParams
from xgboost_ray.tests.utils import create_parquet_in_tempdir

####
# Run `create_test_data.py` first to create a large fake data set.
# Alternatively, run with `--smoke-test` to create an ephemeral small fake
# data set.
####


def main(fname, num_actors=2):
    dtrain = RayDMatrix(
        os.path.abspath(fname), label="labels", ignore=["partition"])

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
        ray_params=RayParams(max_actor_restarts=1, num_actors=num_actors),
        num_boost_round=10,
        evals=[(dtrain, "train")])
    taken = time.time() - start
    print(f"TRAIN TIME TAKEN: {taken:.2f} seconds")

    bst.save_model("test_data.xgb")
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing")
    args = parser.parse_args()

    temp_dir, path = None, None
    if args.smoke_test:
        temp_dir, path = create_parquet_in_tempdir(
            "smoketest.parquet",
            num_rows=1_000,
            num_features=4,
            num_classes=2,
            num_partitions=2)
    else:
        path = os.path.join(os.path.dirname(__file__), "parted.parquet")

    import ray
    ray.init()

    start = time.time()
    main(path)
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")

    if args.smoke_test:
        shutil.rmtree(temp_dir)
