from sklearn import datasets
from sklearn.model_selection import train_test_split

from xgboost_ray import RayDMatrix, train

from ray import tune

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
        num_actors=1,
        verbose_eval=False,
        tune=True)

config = {
         "objective": "binary:logistic",
         "eval_metric": ["logloss", "error"],
         "subsample": tune.uniform(0.5, 1.0),
         "eta": tune.loguniform(1e-4, 1e-1)
     }

analysis = tune.run(
     train_breast_cancer,
     resources_per_trial={"cpu": 0, "extra_cpu": 1},
     config=config,
     num_samples=1)

