import os

from sklearn import datasets

import xgboost as xgb
from xgboost_ray import RayDMatrix, predict

import numpy as np


def main():
    if not os.path.exists("simple.xgb"):
        raise ValueError("Model file not found: `simple.xgb`"
                         "\nFIX THIS by running `python `simple.py` first to "
                         "train the model.")

    # Load dataset
    data, labels = datasets.load_breast_cancer(return_X_y=True)

    dmat_xgb = xgb.DMatrix(data, labels)
    dmat_ray = RayDMatrix(data, labels)

    bst = xgb.Booster(model_file="simple.xgb")

    pred_xgb = bst.predict(dmat_xgb)
    pred_ray = predict(bst, dmat_ray)

    np.testing.assert_array_equal(pred_xgb, pred_ray)
    print(pred_ray)


if __name__ == "__main__":
    main()
