import argparse

import numpy as np
import pandas as pd

import ray

from xgboost_ray import RayDMatrix, train, RayParams


def main(cpus_per_actor, num_actors):
    # Generate dataset
    x = np.repeat(range(8), 16).reshape((32, 4))
    # Even numbers --> 0, odd numbers --> 1
    y = np.tile(np.repeat(range(2), 4), 4)

    # Flip some bits to reduce max accuracy
    bits_to_flip = np.random.choice(32, size=6, replace=False)
    y[bits_to_flip] = 1 - y[bits_to_flip]

    data = pd.DataFrame(x)
    # Ray Datasets require all columns to be string
    data.columns = [str(c) for c in data.columns]
    data["label"] = y

    # There was recent API change - the first clause covers the new
    # and current Ray master API
    if hasattr(ray.data, "from_pandas_refs"):
        # Generate Ray dataset from 4 partitions
        ray_ds = ray.data.from_pandas(data).repartition(num_actors)
    else:
        # Split into 4 partitions
        partitions = [ray.put(part) for part in np.split(data, num_actors)]
        ray_ds = ray.data.from_pandas(partitions)

    train_set = RayDMatrix(ray_ds, "label")

    evals_result = {}
    # Set XGBoost config.
    xgboost_params = {
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    }

    # Train the classifier
    bst = train(
        params=xgboost_params,
        dtrain=train_set,
        evals=[(train_set, "train")],
        evals_result=evals_result,
        ray_params=RayParams(
            max_actor_restarts=0,
            gpus_per_actor=0,
            cpus_per_actor=cpus_per_actor,
            num_actors=num_actors),
        verbose_eval=False,
        num_boost_round=10)

    model_path = "ray_datasets.xgb"
    bst.save_model(model_path)
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for Ray")
    parser.add_argument(
        "--server-address",
        required=False,
        type=str,
        help="Address of the remote server if using Ray Client.")
    parser.add_argument(
        "--cpus-per-actor",
        type=int,
        default=1,
        help="Sets number of CPUs per xgboost training worker.")
    parser.add_argument(
        "--num-actors",
        type=int,
        default=4,
        help="Sets number of xgboost workers to use.")
    parser.add_argument(
        "--smoke-test", action="store_true", default=False, help="gpu")

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init(num_cpus=args.num_actors + 1)
    elif args.server_address:
        ray.util.connect(args.server_address)
    else:
        ray.init(address=args.address)

    main(args.cpus_per_actor, args.num_actors)
