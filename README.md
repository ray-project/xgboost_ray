Distributed XGBoost on Ray
==========================
![Build Status](https://github.com/ray-project/xgboost_ray/workflows/pytest%20on%20push/badge.svg)

This library adds a new backend for XGBoost utilizing the
[distributed computing framework Ray](https://ray.io).

XGBoost-Ray enables multi node and multi GPU 
training with an interface compatible with the usual
XGBoost API. It also integrates with [Ray Tune](#hyperparameter-tuning)
and offers advanced fault tolerance configuration.

All releases are tested on large clusters and workloads.


Installation
------------
You can install the latest XGBoost-Ray release like this:

```bash
pip install xgboost_ray
```

If you'd like to install the latest master, use this command instead:

```bash
pip install git+https://github.com/ray-project/xgboost_ray.git#xgboost_ray
```


Usage
-----
XGBoost-Ray provides a drop-in replacement for XGBoost's `train`
function. To pass data, instead of using `xgb.DMatrix` you will 
have to use `xgboost_ray.RayDMatrix`.

Here is a simplified example (which requires `sklearn`):

**Training:**

```python
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
    ray_params=RayParams(
        num_actors=2,
        cpus_per_actor=1))

bst.save_model("model.xgb")
print("Final training error: {:.4f}".format(
    evals_result["train"]["error"][-1]))
```

**Prediction:**

```python
from xgboost_ray import RayDMatrix, RayParams, predict
from sklearn.datasets import load_breast_cancer
import xgboost as xgb

data, labels = load_breast_cancer(return_X_y=True)

dpred = RayDMatrix(data, labels)

bst = xgb.Booster(model_file="model.xgb")
pred_ray = predict(bst, dpred, ray_params=RayParams(num_actors=2))

print(pred_ray)
```

Data loading
------------

Data is passed to XGBoost-Ray via a `RayDMatrix` object.

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

XGBoost-Ray integrates with [Ray Tune](https://tune.io) to provide distributed hyperparameter tuning for your
distributed XGBoost models. You can run multiple XGBoost-Ray training runs in parallel, each with a different
hyperparameter configuration, and each training run parallelized by itself. All you have to do is move your training
code to a function, and pass the function to `tune.run`. Internally, `train` will detect if `tune` is being used and will
automatically report results to tune.

Example using XGBoost-Ray with Ray Tune:

```python
from xgboost_ray import RayDMatrix, RayParams, train
from sklearn.datasets import load_breast_cancer

num_actors = 4
num_cpus_per_actor = 1

ray_params = RayParams(
    num_actors=num_actors,
    cpus_per_actor=num_cpus_per_actor)

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
```

Also see examples/simple_tune.py for another example.

Resources
---------
By default, XGBoost-Ray tries to determine the number of CPUs
available and distributes them evenly across actors.

In the case of very large clusters or clusters with many different
machine sizes, it makes sense to limit the number of CPUs per actor
by setting the `cpus_per_actor` argument. Consider always
setting this explicitly.

The number of XGBoost actors always has to be set manually with
the `num_actors` argument. 

Memory usage
-------------
XGBoost uses a compute-optimized datastructure, the `DMatrix`,
to hold training data. When converting a dataset to a `DMatrix`,
XGBoost creates intermediate copies and ends up 
holding a complete copy of the full data. The data will be converted
into the local dataformat (on a 64 bit system these are 64 bit floats.)
Depending on the system and original dataset dtype, this matrix can 
thus occupy more memory than the original dataset.

The **peak memory usage** for CPU-based training is at least
**3x** the dataset size (assuming dtype `float32` on a 64bit system) 
plus about **400,000 KiB** for other resources,
like operating system requirements and storing of intermediate
results.

**Example**
- Machine type: AWS m5.xlarge (4 vCPUs, 16 GiB RAM)
- Usable RAM: ~15,350,000 KiB
- Dataset: 1,250,000 rows with 1024 features, dtype float32.
  Total size: 5,000,000 KiB
- XGBoost DMatrix size: ~10,000,000 KiB

This dataset will fit exactly on this node for training.

Note that the DMatrix size might be lower on a 32 bit system.

**GPUs**

Generally, the same memory requirements exist for GPU-based
training. Additionally, the GPU must have enough memory
to hold the dataset. 

In the example above, the GPU must have at least 
10,000,000 KiB (about 9.6 GiB) memory. However, 
empirically we found that using a `DeviceQuantileDMatrix`
seems to show more peak GPU memory usage, possibly 
for intermediate storage when loading data (about 10%).

**Best practices**

In order to reduce peak memory usage, consider the following
suggestions:

- Store data as `float32` or less. More precision is often 
  not needed, and keeping data in a smaller format will
  help reduce peak memory usage for initial data loading.
- Pass the `dtype` when loading data from CSV. Otherwise,
  floating point values will be loaded as `np.float64` 
  per default, increasing peak memory usage by 33%.

Placement Strategies
--------------------
XGBoost-Ray leverages Ray's Placement Group API (https://docs.ray.io/en/master/placement-group.html)
to implement placement strategies for better fault tolerance. 

By default, a SPREAD strategy is used for training, which attempts to spread all of the training workers
across the nodes in a cluster on a best-effort basis. This improves fault tolerance since it minimizes the 
number of worker failures when a node goes down, but comes at a cost of increased inter-node communication
To disable this strategy, set the `USE_SPREAD_STRATEGY` environment variable to 0. If disabled, no
particular placement strategy will be used.

Note that this strategy is used only when `elastic_training` is not used. If `elastic_training` is set to `True`,
no placement strategy is used.

When XGBoost-Ray is used with Ray Tune for hyperparameter tuning, a PACK strategy is used. This strategy
attempts to place all workers for each trial on the same node on a best-effort basis. This means that if a node
goes down, it will be less likely to impact multiple trials.

When placement strategies are used, XGBoost-Ray will wait for 100 seconds for the required resources
to become available, and will fail if the required resources cannot be reserved and the cluster cannot autoscale
to increase the number of resources. You can change the `PLACEMENT_GROUP_TIMEOUT_S` environment variable to modify 
how long this timeout should be. 

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
