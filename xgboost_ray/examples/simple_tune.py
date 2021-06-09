import argparse
import os

import xgboost_ray
from sklearn import datasets
from sklearn.model_selection import train_test_split

import ray
from ray import tune

from xgboost_ray import train, RayDMatrix, RayParams


def train_breast_cancer(config, ray_params):
    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)

    train_set = RayDMatrix(train_x, train_y)
    test_set = RayDMatrix(test_x, test_y)

    evals_result = {}

    bst = train(
        params=config,
        dtrain=train_set,
        evals=[(test_set, "eval")],
        evals_result=evals_result,
        ray_params=ray_params,
        verbose_eval=False,
        num_boost_round=10)

    model_path = "tuned.xgb"
    bst.save_model(model_path)
    print("Final validation error: {:.4f}".format(
        evals_result["eval"]["error"][-1]))


def main(cpus_per_actor, num_actors, num_samples):
    # Set XGBoost config.
    config = {
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "eta": tune.loguniform(1e-4, 1e-1),
        "subsample": tune.uniform(0.5, 1.0),
        "max_depth": tune.randint(1, 9)
    }

    ray_params = RayParams(
        max_actor_restarts=1,
        gpus_per_actor=0,
        cpus_per_actor=cpus_per_actor,
        num_actors=num_actors)

    analysis = tune.run(
        tune.with_parameters(train_breast_cancer, ray_params=ray_params),
        # Use the `get_tune_resources` helper function to set the resources.
        resources_per_trial=ray_params.get_tune_resources(),
        config=config,
        num_samples=num_samples,
        metric="eval-error",
        mode="min")

    # Load the best model checkpoint.
    best_bst = xgboost_ray.tune.load_model(
        os.path.join(analysis.best_logdir, "tuned.xgb"))

    best_bst.save_model("best_model.xgb")

    accuracy = 1. - analysis.best_result["eval-error"]
    print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model total accuracy: {accuracy:.4f}")


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
        help="Sets number of CPUs per XGBoost training worker.")
    parser.add_argument(
        "--num-actors",
        type=int,
        default=1,
        help="Sets number of XGBoost workers to use.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to use for Tune.")
    parser.add_argument("--smoke-test", action="store_true", default=False)

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init(num_cpus=args.num_actors * args.num_samples)
    elif args.server_address:
        ray.util.connect(args.server_address)
    else:
        ray.init(address=args.address)

    main(args.cpus_per_actor, args.num_actors, args.num_samples)
