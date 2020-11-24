import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split

from xgboost_ray import RayDMatrix, train


def create_train_args(params, cpus_per_actor=1, num_actors=1):
    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)

    train_set = RayDMatrix(train_x, train_y)
    test_set = RayDMatrix(test_x, test_y)

    evals_result = {}

    # Train the classifier
    train_args = {
        "params": params,
        "dtrain": train_set,
        "evals": [(test_set, "eval")],
        "evals_result": evals_result,
        "max_actor_restarts": 1,
        "checkpoint_path": "/tmp/checkpoint/",
        "gpus_per_actor": 0,
        "cpus_per_actor": cpus_per_actor,
        "num_actors": num_actors,
        "verbose_eval": False,
        "num_boost_round": 10,
    }

    return train_args


def main(cpus_per_actor, num_actors):
    # Set XGBoost config.
    xgboost_params = {
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    }

    train_args = create_train_args(xgboost_params, cpus_per_actor, num_actors)

    bst = train(**train_args)

    model_path = "simple.xgb"
    bst.save_model(model_path)
    print("Final validation error: {:.4f}".format(
        train_args["evals_result"]["eval"]["error"][-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for Ray")
    parser.add_argument(
        "--cpus-per-actor",
        type=int,
        default=1,
        help="Sets number of CPUs per xgboost training worker.")
    parser.add_argument(
        "--num-actors",
        type=int,
        default=1,
        help="Sets number of xgboost workers to use.")

    args, _ = parser.parse_known_args()

    import ray
    ray.init(address=args.address)

    main(args.cpus_per_actor, args.num_actors)
