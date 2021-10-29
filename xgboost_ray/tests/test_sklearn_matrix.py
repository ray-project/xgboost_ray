import numpy as np
import unittest

import ray
import xgboost as xgb

from sklearn.model_selection import train_test_split

from xgboost_ray.sklearn import (RayXGBClassifier, RayXGBRegressor)
from xgboost_ray.main import RayDMatrix

from xgboost_ray.main import XGBOOST_VERSION_TUPLE

has_label_encoder = (XGBOOST_VERSION_TUPLE >= (1, 0, 0)
                     and XGBOOST_VERSION_TUPLE < (1, 6, 0))


class XGBoostRaySklearnMatrixTest(unittest.TestCase):
    def setUp(self):
        self.seed = 1994
        self.rng = np.random.RandomState(self.seed)
        self.params = {"n_estimators": 10}

    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

    def _init_ray(self):
        if not ray.is_initialized():
            ray.init(num_cpus=4)

    @unittest.skipIf(not has_label_encoder,
                     f"not supported in xgb version {xgb.__version__}")
    def testClassifierLabelEncoder(self, n_class=2):
        self._init_ray()

        from sklearn.datasets import load_digits

        digits = load_digits(n_class=n_class)
        y = digits["target"]
        X = digits["data"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5)

        train_matrix = RayDMatrix(X_train, y_train)
        test_matrix = RayDMatrix(X_test, y_test)

        with self.assertRaisesRegex(Exception, "use_label_encoder"):
            RayXGBClassifier(
                use_label_encoder=True, **self.params).fit(train_matrix, None)

        with self.assertRaisesRegex(Exception, "num_class"):
            RayXGBClassifier(
                use_label_encoder=False, **self.params).fit(
                    train_matrix, None)

        with self.assertRaisesRegex(Exception, r"must be \(RayDMatrix, str\)"):
            RayXGBClassifier(
                use_label_encoder=False, **self.params).fit(
                    train_matrix, None, eval_set=[(X_test, y_test)])

        with self.assertRaisesRegex(Exception,
                                    r"must be \(array_like, array_like\)"):
            RayXGBClassifier(
                use_label_encoder=False, **self.params).fit(
                    X_train, y_train, eval_set=[(test_matrix, "eval")])

        RayXGBClassifier(
            use_label_encoder=False, num_class=n_class, **self.params).fit(
                train_matrix, None)

        clf = RayXGBClassifier(
            use_label_encoder=False, num_class=n_class, **self.params).fit(
                train_matrix, None, eval_set=[(test_matrix, "eval")])

        clf.predict(test_matrix)
        clf.predict_proba(test_matrix)

    @unittest.skipIf(not has_label_encoder,
                     f"not supported in xgb version {xgb.__version__}")
    def testClassifierMulticlassLabelEncoder(self):
        self.testClassifierLabelEncoder(n_class=3)

    @unittest.skipIf(has_label_encoder,
                     f"not supported in xgb version {xgb.__version__}")
    def testClassifierNoLabelEncoder(self, n_class=2):
        self._init_ray()

        from sklearn.datasets import load_digits

        digits = load_digits(n_class=n_class)
        y = digits["target"]
        X = digits["data"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5)

        train_matrix = RayDMatrix(X_train, y_train)
        test_matrix = RayDMatrix(X_test, y_test)

        with self.assertRaisesRegex(Exception, "num_class"):
            RayXGBClassifier(**self.params).fit(train_matrix, None)

        with self.assertRaisesRegex(Exception, r"must be \(RayDMatrix, str\)"):
            RayXGBClassifier(**self.params).fit(
                train_matrix, None, eval_set=[(X_test, y_test)])

        with self.assertRaisesRegex(Exception,
                                    r"must be \(array_like, array_like\)"):
            RayXGBClassifier(**self.params).fit(
                X_train, y_train, eval_set=[(test_matrix, "eval")])

        RayXGBClassifier(
            num_class=n_class, **self.params).fit(train_matrix, None)

        clf = RayXGBClassifier(
            num_class=n_class, **self.params).fit(
                train_matrix, None, eval_set=[(test_matrix, "eval")])

        clf.predict(test_matrix)
        clf.predict_proba(test_matrix)

    @unittest.skipIf(has_label_encoder,
                     f"not supported in xgb version {xgb.__version__}")
    def testClassifierMulticlassNoLabelEncoder(self):
        self.testClassifierNoLabelEncoder(n_class=3)

    def testRegressor(self):
        self._init_ray()

        from sklearn.datasets import load_boston

        boston = load_boston()
        y = boston["target"]
        X = boston["data"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5)

        train_matrix = RayDMatrix(X_train, y_train)
        test_matrix = RayDMatrix(X_test, y_test)

        with self.assertRaisesRegex(Exception, r"must be \(RayDMatrix, str\)"):
            RayXGBRegressor(**self.params).fit(
                train_matrix, None, eval_set=[(X_test, y_test)])

        with self.assertRaisesRegex(Exception,
                                    r"must be \(array_like, array_like\)"):
            RayXGBRegressor(**self.params).fit(
                X_train, y_train, eval_set=[(test_matrix, "eval")])

        RayXGBRegressor(**self.params).fit(train_matrix, None)

        reg = RayXGBRegressor(**self.params).fit(
            train_matrix, None, eval_set=[(test_matrix, "eval")])

        reg.predict(test_matrix)
