from sklearn import datasets
from sklearn.model_selection import train_test_split

from xgboost_ray import RayDMatrix, train

from ray import tune
from ray.tune.integration.xgboost import TuneReportCallback

def train_breast_cancer(config):
    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=0.25)

    train_set = RayDMatrix(train_x, train_y)
    test_set = RayDMatrix(test_x, test_y)

    evals_result = {}

    # Train the classifier
    train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        evals_result=evals_result,
        max_actor_restarts=1,
        gpus_per_actor=0,
        cpus_per_actor=1,
        num_actors=4,
        verbose_eval=False,
        callbacks=[TuneReportCallback()])

    error = evals_result["eval"]["error"][-1]
    accuracy = 1. - error
    tune.report(mean_accuracy=accuracy, done=True)

config = {
         "objective": "binary:logistic",
         "eval_metric": ["logloss", "error"],
         #"max_depth": tune.randint(1, 9),
         #"min_child_weight": tune.choice([1, 2, 3]),
         "subsample": tune.uniform(0.5, 1.0),
         "eta": tune.loguniform(1e-4, 1e-1)
     }

analysis = tune.run(
     train_breast_cancer,
     resources_per_trial={"cpu": 0, "extra_cpu": 4},
     config=config,
     num_samples=4)

