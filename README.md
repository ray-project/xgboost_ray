Distributed XGBoost on Ray
==========================

This library adds a new backend for XGBoost utilizing the
[distributed computing framework Ray](https://ray.io).

Please note that this is an early version and both the API and
the behavior can change without prior notice.

We'll switch to a release-based development process once the
implementation has all features for first real world use cases.

Installation
------------
You can install `xgboost_ray` like this:

```
git clone https://github.com/ray-project/xgboost_ray.git
cd xgboost_ray
pip install -e .
```

Usage
-----
`xgboost_ray` provides a drop-in replacement for XGBoost's `train`
function. To pass data, instead of using `xgb.DMatrix` you will 
have to use `xgboost_ray.RayDMatrix`.

Here is a simplified example:

```python
from xgboost_ray import RayDMatrix, train

train_x, train_y = None, None  # Load data here
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
    num_actors=2,
    cpus_per_actor=1)

bst.save_model("model.xgb")
print("Final training error: {:.4f}".format(
    evals_result["train"]["error"][-1]))
```

Data loading
------------

Data is passed to `xgboost_ray` via a `RayDMatrix` object.

The `RayDMatrix` lazy loads data and stores it sharded in the
Ray object store. The Ray XGBoost actors then access these
shards to run their training on. 

A `RayDMatrix` support various data and file types, like
Pandas DataFrames, Numpy Arrays, CSV files and Parquet files.

Example loading multiple parquet files:

```python
import glob    
from xgboost_ray import RayDMatrix, RayFileType

# We can also pass a list of files
path = list(sorted(glob.glob("/data/nyc-taxi/*/*/*.parquet")))

# This argument will be passed to `pd.read_parquet()`
columns = [
    "passenger_count",
    "trip_distance", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "fare_amount", "extra", "mta_tax", "tip_amount",
    "tolls_amount", "total_amount"
]

dtrain = RayDMatrix(
    path, 
    label="passenger_count",  # Will select this column as the label
    columns=columns, 
    filetype=RayFileType.PARQUET)
```

Hyperparameter Tuning
---------------------

`xgboost_ray` integrates with [Ray Tune](https://tune.io) to provide distributed hyperparameter tuning for your
distributed XGBoost models. You can run multiple `xgboost_ray` training runs in parallel, each with a different
hyperparameter configuration, and each training run parallelized by itself. All you have to do is move your training
code to a function, and pass the function to `tune.run`. Internally, `train` will detect if `tune` is being used and will
automatically report results to tune.

Example using `xgboost_ray` with Tune:

```python
from xgboost_ray import RayDMatrix, train

num_actors = 4
num_cpus_per_actor = 4

def train_model(config):
    train_x, train_y = None, None  # Load data here
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
        num_actors=num_actors,
        cpus_per_actor=num_cpus_per_actor)
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

# Make sure to specify how many actors each training run will create via the "extra_cpu" field.
tune.run(train_model, config=config, resources_per_trial={"cpu": 1, "extra_cpu": num_actors*num_cpus_per_actor})
```

Also see examples/simple_tune.py for another example.

Resources
---------
By default, `xgboost_ray` tries to determine the number of CPUs
available and distributes them evenly across actors.

In the case of very large clusters or clusters with many different
machine sizes, it makes sense to limit the number of CPUs per actor
by setting the `cpus_per_actor` argument. Consider always
setting this explicitly.

The number of XGBoost actors always has to be set manually with
the `num_actors` argument. 

More examples
-------------

Fore complete end to end examples, please have a look at 
the [examples folder](examples/):

* [Simple sklearn breastcancer dataset example](examples/simple.py) (requires `sklearn`)
* [HIGGS classification example](examples/higgs.py) 
([download dataset (2.6 GB)](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz))
* [HIGGS classification example with Parquet](examples/higgs_parquet.py) (uses the same dataset) 
* [Test data classification](examples/train_on_test_data.py) (uses a self-generated dataset) 


Resources
---------
* [Ray community slack](https://forms.gle/9TSdDYUgxYs8SA9e8)
