import argparse
import os

from ray import tune

import xgboost as xgb

from xgboost_ray import hyperparameter_search

from examples.simple import create_train_args


def train_breast_cancer(config, cpus_per_actor=1, num_actors=1):

    train_args = create_train_args(params=config,
                                   cpus_per_actor=cpus_per_actor,
                                   num_actors=num_actors)

    bst = hyperparameter_search(**train_args, metrics=["eval-error"])

    model_path = "simple.xgb"
    with tune.checkpoint_dir(step=0) as checkpoint_dir:
        model_path = checkpoint_dir + "/" + model_path
    bst.save_model(model_path)
    print("Final validation error: {:.4f}".format(
        train_args["evals_result"]["eval"]["error"][-1]))

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
    best_bst.load_model(
        os.path.join(analysis.best_checkpoint, "simple.xgb"))
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
        help="Sets number of CPUs per xgboost training worker.")
    parser.add_argument(
        "--num-actors",
        type=int,
        default=1,
        help="Sets number of xgboost workers to use.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number "
        "of "
        "samples "
        "to use "
        "for Tune.")

    args, _ = parser.parse_known_args()

    import ray
    ray.init(address=args.address)
    main(args.cpus_per_actor, args.num_actors, args.num_samples)
