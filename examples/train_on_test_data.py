import os
import time

from xgboost_ray import train, RayDMatrix


def main():
    # Run `create_test_data.py` first to create fake data.
    fname = "parted.parquet"

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
        max_actor_restarts=1,
        num_boost_round=100,
        evals=[(dtrain, "train")])
    taken = time.time() - start
    print(f"TRAIN TIME TAKEN: {taken:.2f} seconds")

    bst.save_model("test_data.xgb")
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))


if __name__ == "__main__":
    import ray
    ray.init()

    start = time.time()
    main()
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
