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

Please consider always explicitly setting the number of actors and
the number of CPUs per actor.

Fore complete end to end examples, please have a look at 
the [examples folder](examples/):

* [Simple sklearn breastcancer dataset example](examples/simple.py) (requires `sklearn`)
* [HIGGS classification example](examples/higgs.py) 
([download dataset (2.6 GB)](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz))

Resources
---------
* [Ray community slack](https://forms.gle/9TSdDYUgxYs8SA9e8)
