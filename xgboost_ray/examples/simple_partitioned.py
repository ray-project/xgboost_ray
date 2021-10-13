import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np

import ray

from xgboost_ray import RayDMatrix, train, RayParams

nc = 31


@ray.remote
class AnActor:
    """We mimic a distributed DF by having several actors create
       data which form the global DF.
    """

    @ray.method(num_returns=2)
    def genData(self, rank, nranks, nrows):
        """Generate global dataset and cut out local piece.
           In real life each actor would of course directly create local data.
        """
        # Load dataset
        data, labels = datasets.load_breast_cancer(return_X_y=True)
        # Split into train and test set
        train_x, _, train_y, _ = train_test_split(data, labels, test_size=0.25)
        train_y = train_y.reshape((train_y.shape[0], 1))
        train = np.hstack([train_x, train_y])
        assert nrows <= train.shape[0]
        assert nc == train.shape[1]
        sz = nrows // nranks
        return train[sz * rank:sz * (rank + 1)], ray.util.get_node_ip_address()


class Parted:
    """Class exposing __partitioned__
    """

    def __init__(self, parted):
        self.__partitioned__ = parted


def main(cpus_per_actor, num_actors):
    nr = 424
    actors = [AnActor.remote() for _ in range(num_actors)]
    parts = [
        actors[i].genData.remote(i, num_actors, nr) for i in range(num_actors)
    ]
    rowsperpart = nr // num_actors
    nr = rowsperpart * num_actors
    parted = Parted({
        "shape": (nr, nc),
        "partition_tiling": (num_actors, 1),
        "get": lambda x: ray.get(x),
        "partitions": {(i, 0): {
            "start": (i * rowsperpart, 0),
            "shape": (rowsperpart, nc),
            "data": parts[i][0],
            "location": [ray.get(parts[i][1])],
        }
                       for i in range(num_actors)}
    })

    yl = nc - 1
    # Let's create DMatrix from our __partitioned__ structure
    train_set = RayDMatrix(parted, f"f{yl}")

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

    model_path = "partitioned.xgb"
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

    if not ray.is_initialized():
        if args.smoke_test:
            ray.init(num_cpus=args.num_actors + 1)
        elif args.server_address:
            ray.util.connect(args.server_address)
        else:
            ray.init(address=args.address)

    main(args.cpus_per_actor, args.num_actors)
