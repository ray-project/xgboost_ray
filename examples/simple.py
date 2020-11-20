import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split

from xgboost_ray import RayDMatrix, train

def train_breast_cancer(config, cpus_per_actor=1, num_actors=1,
                        use_tune=False):
    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)

    train_set = RayDMatrix(train_x, train_y)
    test_set = RayDMatrix(test_x, test_y)

    evals_result = {}


    # Train the classifier
    bst = train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        evals_result=evals_result,
        max_actor_restarts=1,
        gpus_per_actor=0,
        cpus_per_actor=cpus_per_actor,
        num_actors=num_actors,
        verbose_eval=False,
        tune=use_tune)

    if not use_tune:
        bst.save_model("simple.xgb")
        print("Final validation error: {:.4f}".format(
            evals_result["eval"]["error"][-1]))


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
        "--tune", action="store_true", default=False, help="Tune training")

    args, _ = parser.parse_known_args()

    import ray
    ray.init(address=args.address)
    # Set XGBoost config.
    config = {
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": 3,
    }

    if args.tune:
        from ray import tune
        # Set config for tune.
        config.update({
            "eta": tune.loguniform(1e-4, 1e-1),
            "subsample": tune.uniform(0.5, 1.0)
        })
        tune.run(tune.with_parameters(train_breast_cancer,
                                      cpus_per_actor=args.cpus_per_actor,
                                      num_actors=args.num_actors,
                                      use_tune=True), resources_per_trial={
            "cpu": 0, "extra_cpu": args.cpus_per_actor*args.num_actors},
            config=config, num_samples=4)
    else:
        train_breast_cancer(config, cpus_per_actor=args.cpus_per_actor,
                            num_actors=args.num_actors, use_tune=False)
