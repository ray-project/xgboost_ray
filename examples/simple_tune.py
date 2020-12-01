import argparse
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ray import tune

from xgboost_ray import train, RayDMatrix, RayParams


def train_breast_cancer(config, cpus_per_actor=1, num_actors=1):
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
        ray_params=RayParams(
            max_actor_restarts=1,
            gpus_per_actor=0,
            cpus_per_actor=cpus_per_actor,
            num_actors=num_actors),
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

    analysis = tune.run(
        tune.with_parameters(
            train_breast_cancer,
            cpus_per_actor=cpus_per_actor,
            num_actors=num_actors),
        # extra_cpu is used if the trainable creates additional remote actors.
        # https://docs.ray.io/en/master/tune/api_docs/trainable.html#advanced-resource-allocation
        resources_per_trial={
            "cpu": 1,
            "extra_cpu": cpus_per_actor * num_actors
        },
        config=config,
        num_samples=num_samples,
        metric="eval-error",
        mode="min")

    # Load the best model checkpoint
    best_bst = xgb.Booster()
    best_bst.load_model(os.path.join(analysis.best_logdir, "tuned.xgb"))
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

    args, _ = parser.parse_known_args()

    import ray
    ray.init(address=args.address)
    main(args.cpus_per_actor, args.num_actors, args.num_samples)
