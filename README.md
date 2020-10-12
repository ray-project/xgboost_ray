Distributed XGBoost on Ray
==========================

This library adds a new backend for XGBoost utilizing the
[distributed computing framework Ray](https://ray.io).

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

bst, evals = train(
    {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    },
    train_set,
    evals=[(train_set, "train")],
    verbose_eval=False)

bst.save_model("model.xgb")
print("Final training error: {:.4f}".format(evals["train"]["error"][-1]))
```

Fore complete end to end examples, please have a look at 
the [examples folder](examples/):

* [Simple sklearn breastcancer dataset example](examples/simple.py)
* [HIGGS classification example](examples/higgs.py)

Resources
---------
* [Ray community slack](https://forms.gle/9TSdDYUgxYs8SA9e8)
