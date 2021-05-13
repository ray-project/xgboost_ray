# flake8: noqa E501


def readme_simple():
    from xgboost_ray import RayDMatrix, RayParams, train
    from sklearn.datasets import load_breast_cancer

    train_x, train_y = load_breast_cancer(return_X_y=True)
    train_set = RayDMatrix(train_x, train_y)

    evals_result = {}
    bst = train(
        {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
        },
        train_set,
        evals_result=evals_result,
        evals=[(train_set, "train")],
        verbose_eval=False,
        ray_params=RayParams(num_actors=2, cpus_per_actor=1))

    bst.save_model("model.xgb")
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))


def readme_predict():
    from xgboost_ray import RayDMatrix, RayParams, predict
    from sklearn.datasets import load_breast_cancer
    import xgboost as xgb

    data, labels = load_breast_cancer(return_X_y=True)

    dpred = RayDMatrix(data, labels)

    bst = xgb.Booster(model_file="model.xgb")
    pred_ray = predict(bst, dpred, ray_params=RayParams(num_actors=2))

    print(pred_ray)


def readme_tune():
    from xgboost_ray import RayDMatrix, RayParams, train
    from sklearn.datasets import load_breast_cancer

    num_actors = 4
    num_cpus_per_actor = 1

    ray_params = RayParams(
        num_actors=num_actors, cpus_per_actor=num_cpus_per_actor)

    def train_model(config):
        train_x, train_y = load_breast_cancer(return_X_y=True)
        train_set = RayDMatrix(train_x, train_y)

        evals_result = {}
        bst = train(
            params=config,
            dtrain=train_set,
            evals_result=evals_result,
            evals=[(train_set, "train")],
            verbose_eval=False,
            ray_params=ray_params)
        bst.save_model("model.xgb")

    from ray import tune

    # Specify the hyperparameter search space.
    config = {
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "eta": tune.loguniform(1e-4, 1e-1),
        "subsample": tune.uniform(0.5, 1.0),
        "max_depth": tune.randint(1, 9)
    }

    # Make sure to use the `get_tune_resources` method to set the `resources_per_trial`
    analysis = tune.run(
        train_model,
        config=config,
        metric="train-error",
        mode="min",
        num_samples=4,
        resources_per_trial=ray_params.get_tune_resources())
    print("Best hyperparameters", analysis.best_config)


if __name__ == "__main__":
    import ray

    ray.init(num_cpus=5)

    print("Readme: Simple example")
    readme_simple()
    readme_predict()
    try:
        print("Readme: Ray Tune example")
        readme_tune()
    except ImportError:
        print("Ray Tune not installed.")
