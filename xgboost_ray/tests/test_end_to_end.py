import numpy as np
import unittest
import xgboost as xgb

import ray

from xgboost_ray import RayParams, train, RayDMatrix


class XGBoostRayEndToEndTest(unittest.TestCase):
    """In this test suite we validate Ray-XGBoost multi class prediction.

    First, we validate that XGBoost is able to achieve 100% accuracy on
    a simple training task.

    Then we split the dataset into two halves. These halves don't have access
    to all relevant data, so overfit on their respective data. I.e. the first
    half always predicts feature 2 -> label 2, while the second half always
    predicts feature 2 -> label 3.

    We then train using Ray XGBoost. Again both halves will be trained
    separately, but because of Rabit's allreduce, they should end up being
    able to achieve 100% accuracy, again."""

    def setUp(self):
        repeat = 8  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        self.y = np.array([0, 1, 2, 3] * repeat)

        self.params = {
            "booster": "gbtree",
            "nthread": 1,
            "max_depth": 2,
            "objective": "multi:softmax",
            "num_class": 4
        }

    def testSingleTraining(self):
        """Test that XGBoost learns to predict full matrix"""
        dtrain = xgb.DMatrix(self.x, self.y)
        bst = xgb.train(self.params, dtrain, num_boost_round=2)

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))

    def testHalfTraining(self):
        """Test that XGBoost learns to predict half matrices individually"""
        x_first = self.x[::2]
        y_first = self.y[::2]

        x_second = self.x[1::2]
        y_second = self.y[1::2]

        # Test case: The first model only sees feature 2 --> label 2
        # and the second model only sees feature 2 --> label 3
        test_X = xgb.DMatrix(np.array([[0, 0, 1, 1], [0, 0, 1, 0]]))
        test_y_first = [2, 2]
        test_y_second = [3, 3]

        # First half
        dtrain = xgb.DMatrix(x_first, y_first)
        bst = xgb.train(self.params, dtrain, num_boost_round=2)

        x_mat = xgb.DMatrix(x_first)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(y_first), list(pred_y))

        pred_test = bst.predict(test_X)
        self.assertSequenceEqual(test_y_first, list(pred_test))

        # Second half
        dtrain = xgb.DMatrix(x_second, y_second)
        bst = xgb.train(self.params, dtrain, num_boost_round=2)

        x_mat = xgb.DMatrix(x_second)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(y_second), list(pred_y))

        pred_test = bst.predict(test_X)
        self.assertSequenceEqual(test_y_second, list(pred_test))

    def testJointTraining(self):
        """Train with Ray. The data will be split, but the trees
        should be combined together and find the true model."""
        ray.init(num_cpus=2, num_gpus=0)

        bst = train(
            self.params,
            RayDMatrix(self.x, self.y),
            ray_params=RayParams(num_actors=2))

        x_mat = xgb.DMatrix(self.x)
        pred_y = bst.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
