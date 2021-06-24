"""Copied almost verbatim from https://github.com/dmlc/xgboost/blob/a5c852660b1056204aa2e0cbfcd5b4ecfbf31adf/tests/python/test_with_sklearn.py
in order to ensure 1:1 coverage, with minimal modifications.
Some tests were disabled due to not being applicable for a
distributed setting."""  # noqa: E501

# Copyright 2021 by XGBoost Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File based on:
# https://github.com/dmlc/xgboost/blob/a5c852660b1056204aa2e0cbfcd5b4ecfbf31adf/tests/python/test_with_sklearn.py

# License:
# https://github.com/dmlc/xgboost/blob/a5c852660b1056204aa2e0cbfcd5b4ecfbf31adf/LICENSE

# import collections
# import importlib.util
import numpy as np
import xgboost as xgb
import unittest

# import io
# from contextlib import redirect_stdout, redirect_stderr
import tempfile
import os
import shutil
import json

import ray

from xgboost_ray.sklearn import (RayXGBClassifier, RayXGBRegressor,
                                 RayXGBRFClassifier, RayXGBRFRegressor,
                                 RayXGBRanker)

from xgboost_ray.main import XGBOOST_VERSION_TUPLE
from xgboost_ray.matrix import RayShardingMode


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


def softprob_obj(classes):
    def objective(labels, predt):
        rows = labels.shape[0]
        grad = np.zeros((rows, classes), dtype=float)
        hess = np.zeros((rows, classes), dtype=float)
        eps = 1e-6
        for r in range(predt.shape[0]):
            target = labels[r]
            p = softmax(predt[r, :])
            for c in range(predt.shape[1]):
                assert target >= 0 or target <= classes
                g = p[c] - 1.0 if c == target else p[c]
                h = max((2.0 * p[c] * (1.0 - p[c])).item(), eps)
                grad[r, c] = g
                hess[r, c] = h

        grad = grad.reshape((rows * classes, 1))
        hess = hess.reshape((rows * classes, 1))
        return grad, hess

    return objective


class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp()"""

    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)


class XGBoostRaySklearnTest(unittest.TestCase):
    def setUp(self):
        self.seed = 1994
        self.rng = np.random.RandomState(self.seed)

    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

    def _init_ray(self):
        if not ray.is_initialized():
            ray.init(num_cpus=4)

    def run_binary_classification(self, cls, ray_dmatrix_params=None):
        self._init_ray()

        from sklearn.datasets import load_digits
        from sklearn.model_selection import KFold

        digits = load_digits(n_class=2)
        y = digits["target"]
        X = digits["data"]
        kf = KFold(n_splits=2, shuffle=True, random_state=self.rng)

        for train_index, test_index in kf.split(X, y):
            clf = cls(random_state=42)
            xgb_model = clf.fit(
                X[train_index],
                y[train_index],
                eval_metric=["auc", "logloss"],
                ray_dmatrix_params=ray_dmatrix_params,
            )
            preds = xgb_model.predict(
                X[test_index], ray_dmatrix_params=ray_dmatrix_params)
            labels = y[test_index]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1

    def test_binary_classification(self):
        self.run_binary_classification(RayXGBClassifier)

    def test_binary_classification_dmatrix_params(self):
        self.run_binary_classification(
            RayXGBClassifier,
            ray_dmatrix_params={"sharding": RayShardingMode.BATCH})

    # ray: added for legacy CI test
    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 0, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_binary_rf_classification(self):
        self.run_binary_classification(RayXGBRFClassifier)

    def test_multiclass_classification(self):
        self._init_ray()

        from sklearn.datasets import load_iris
        from sklearn.model_selection import KFold

        def check_pred(preds, labels, output_margin):
            if output_margin:
                err = sum(1 for i in range(len(preds))
                          if preds[i].argmax() != labels[i]) / float(
                              len(preds))
            else:
                err = sum(1 for i in range(len(preds))
                          if preds[i] != labels[i]) / float(len(preds))
            assert err < 0.4

        iris = load_iris()
        y = iris["target"]
        X = iris["data"]
        kf = KFold(n_splits=2, shuffle=True, random_state=self.rng)
        for train_index, test_index in kf.split(X, y):
            xgb_model = RayXGBClassifier().fit(X[train_index], y[train_index])
            if hasattr(xgb_model.get_booster(), "num_boosted_rounds"):
                assert (xgb_model.get_booster().num_boosted_rounds() ==
                        xgb_model.n_estimators)
            preds = xgb_model.predict(X[test_index])
            # test other params in XGBClassifier().fit
            preds2 = xgb_model.predict(
                X[test_index], output_margin=True, ntree_limit=3)
            preds3 = xgb_model.predict(
                X[test_index], output_margin=True, ntree_limit=0)
            preds4 = xgb_model.predict(
                X[test_index], output_margin=False, ntree_limit=3)
            labels = y[test_index]

            check_pred(preds, labels, output_margin=False)
            check_pred(preds2, labels, output_margin=True)
            check_pred(preds3, labels, output_margin=True)
            check_pred(preds4, labels, output_margin=False)

        cls = RayXGBClassifier(n_estimators=4).fit(X, y)
        assert cls.n_classes_ == 3
        proba = cls.predict_proba(X)
        assert proba.shape[0] == X.shape[0]
        assert proba.shape[1] == cls.n_classes_

        # custom objective, the default is multi:softprob
        # so no transformation is required.
        cls = RayXGBClassifier(
            n_estimators=4, objective=softprob_obj(3)).fit(X, y)
        proba = cls.predict_proba(X)
        assert proba.shape[0] == X.shape[0]
        assert proba.shape[1] == cls.n_classes_

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 4, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_best_ntree_limit(self):
        self._init_ray()

        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)

        def train(booster, forest):
            rounds = 4
            cls = RayXGBClassifier(
                n_estimators=rounds, num_parallel_tree=forest,
                booster=booster).fit(
                    X, y, eval_set=[(X, y)], early_stopping_rounds=3)

            if forest:
                assert cls.best_ntree_limit == rounds * forest
            else:
                assert cls.best_ntree_limit == 0

            # best_ntree_limit is used by default,
            # assert that under gblinear it's
            # automatically ignored due to being 0.
            cls.predict(X)

        num_parallel_tree = 4
        train("gbtree", num_parallel_tree)
        train("dart", num_parallel_tree)
        train("gblinear", None)

    def test_stacking_regression(self):
        self._init_ray()

        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes
        from sklearn.linear_model import RidgeCV
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import StackingRegressor

        X, y = load_diabetes(return_X_y=True)
        estimators = [
            ("gbm", RayXGBRegressor(objective="reg:squarederror")),
            ("lr", RidgeCV()),
        ]
        reg = StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor(
                n_estimators=10, random_state=42),
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        reg.fit(X_train, y_train).score(X_test, y_test)

    def test_stacking_classification(self):
        self._init_ray()

        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_iris
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import StackingClassifier

        X, y = load_iris(return_X_y=True)
        estimators = [
            ("gbm", RayXGBClassifier()),
            (
                "svr",
                make_pipeline(StandardScaler(), LinearSVC(random_state=42)),
            ),
        ]
        clf = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)
        clf.fit(X_train, y_train).score(X_test, y_test)

    # exact tree method doesn't support distributed training
    # def test_feature_importances_weight(self)

    # def test_feature_importances_gain(self)

    def test_select_feature(self):
        self._init_ray()

        from sklearn.datasets import load_digits
        from sklearn.feature_selection import SelectFromModel

        digits = load_digits(n_class=2)
        y = digits["target"]
        X = digits["data"]
        cls = RayXGBClassifier()
        cls.fit(X, y)
        selector = SelectFromModel(cls, prefit=True, max_features=1)
        X_selected = selector.transform(X)
        assert X_selected.shape[1] == 1

    def test_num_parallel_tree(self):
        self._init_ray()

        from sklearn.datasets import load_boston

        reg = RayXGBRegressor(
            n_estimators=4, num_parallel_tree=4, tree_method="hist")
        boston = load_boston()
        bst = reg.fit(X=boston["data"], y=boston["target"])
        dump = bst.get_booster().get_dump(dump_format="json")
        assert len(dump) == 16

        if xgb.__version__ != "0.90":
            reg = RayXGBRFRegressor(n_estimators=4)
            bst = reg.fit(X=boston["data"], y=boston["target"])
            dump = bst.get_booster().get_dump(dump_format="json")
            assert len(dump) == 4

            config = json.loads(bst.get_booster().save_config())
            assert (int(config["learner"]["gradient_booster"][
                "gbtree_train_param"]["num_parallel_tree"]) == 4)

    def test_boston_housing_regression(self):
        self._init_ray()

        from sklearn.metrics import mean_squared_error
        from sklearn.datasets import load_boston
        from sklearn.model_selection import KFold

        boston = load_boston()
        y = boston["target"]
        X = boston["data"]
        kf = KFold(n_splits=2, shuffle=True, random_state=self.rng)
        for train_index, test_index in kf.split(X, y):
            xgb_model = RayXGBRegressor().fit(X[train_index], y[train_index])

            preds = xgb_model.predict(X[test_index])
            # test other params in XGBRegressor().fit
            preds2 = xgb_model.predict(
                X[test_index], output_margin=True, ntree_limit=3)
            preds3 = xgb_model.predict(
                X[test_index], output_margin=True, ntree_limit=0)
            preds4 = xgb_model.predict(
                X[test_index], output_margin=False, ntree_limit=3)
            labels = y[test_index]

            assert mean_squared_error(preds, labels) < 25
            assert mean_squared_error(preds2, labels) < 350
            assert mean_squared_error(preds3, labels) < 25
            assert mean_squared_error(preds4, labels) < 350

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 0, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def run_boston_housing_rf_regression(self, tree_method):
        from sklearn.metrics import mean_squared_error
        from sklearn.datasets import load_boston
        from sklearn.model_selection import KFold

        X, y = load_boston(return_X_y=True)
        kf = KFold(n_splits=2, shuffle=True, random_state=self.rng)
        for train_index, test_index in kf.split(X, y):
            xgb_model = RayXGBRFRegressor(
                random_state=42, tree_method=tree_method).fit(
                    X[train_index], y[train_index])
            preds = xgb_model.predict(X[test_index])
            labels = y[test_index]
            assert mean_squared_error(preds, labels) < 35

    def test_boston_housing_rf_regression(self):
        self._init_ray()

        self.run_boston_housing_rf_regression("hist")

    def test_parameter_tuning(self):
        self._init_ray()

        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_boston

        boston = load_boston()
        y = boston["target"]
        X = boston["data"]
        xgb_model = RayXGBRegressor(learning_rate=0.1)
        clf = GridSearchCV(
            xgb_model,
            {
                "max_depth": [2, 4, 6],
                "n_estimators": [50, 100, 200]
            },
            cv=3,
            verbose=1,
        )
        clf.fit(X, y)
        assert clf.best_score_ < 0.7
        assert clf.best_params_ == {"n_estimators": 100, "max_depth": 4}

    def test_regression_with_custom_objective(self):
        self._init_ray()

        from sklearn.metrics import mean_squared_error
        from sklearn.datasets import load_boston
        from sklearn.model_selection import KFold

        def objective_ls(y_true, y_pred):
            grad = y_pred - y_true
            hess = np.ones(len(y_true))
            return grad, hess

        boston = load_boston()
        y = boston["target"]
        X = boston["data"]
        kf = KFold(n_splits=2, shuffle=True, random_state=self.rng)
        for train_index, test_index in kf.split(X, y):
            xgb_model = RayXGBRegressor(objective=objective_ls).fit(
                X[train_index], y[train_index])
            preds = xgb_model.predict(X[test_index])
            labels = y[test_index]
        assert mean_squared_error(preds, labels) < 25

        # Test that the custom objective function is actually used
        class XGBCustomObjectiveException(Exception):
            pass

        def dummy_objective(y_true, y_pred):
            raise XGBCustomObjectiveException()

        xgb_model = RayXGBRegressor(objective=dummy_objective)
        # TODO figure out how to assertRaises XGBCustomObjectiveException
        with self.assertRaises(RuntimeError):
            xgb_model.fit(X, y)

    def test_classification_with_custom_objective(self):
        self._init_ray()

        from sklearn.datasets import load_digits
        from sklearn.model_selection import KFold

        def logregobj(y_true, y_pred):
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            grad = y_pred - y_true
            hess = y_pred * (1.0 - y_pred)
            return grad, hess

        digits = load_digits(n_class=2)
        y = digits["target"]
        X = digits["data"]
        kf = KFold(n_splits=2, shuffle=True, random_state=self.rng)
        for train_index, test_index in kf.split(X, y):
            xgb_model = RayXGBClassifier(objective=logregobj)
            xgb_model.fit(X[train_index], y[train_index])
            preds = xgb_model.predict(X[test_index])
            labels = y[test_index]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1

        # Test that the custom objective function is actually used
        class XGBCustomObjectiveException(Exception):
            pass

        def dummy_objective(y_true, y_preds):
            raise XGBCustomObjectiveException()

        xgb_model = RayXGBClassifier(objective=dummy_objective)
        # TODO figure out how to assertRaises XGBCustomObjectiveException
        with self.assertRaises(RuntimeError):
            xgb_model.fit(X, y)

        # cls = RayXGBClassifier(use_label_encoder=False)
        # cls.fit(X, y)

        # is_called = [False]

        # def wrapped(y, p):
        #     is_called[0] = True
        #     return logregobj(y, p)

        # cls.set_params(objective=wrapped)
        # cls.predict(X)  # no throw
        # cls.fit(X, y)

        # assert is_called[0]

    def test_sklearn_api(self):
        self._init_ray()

        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        iris = load_iris()
        tr_d, te_d, tr_l, te_l = train_test_split(
            iris.data, iris.target, train_size=120, test_size=0.2)

        classifier = RayXGBClassifier(
            booster="gbtree", n_estimators=10, random_state=self.seed)
        classifier.fit(tr_d, tr_l)

        preds = classifier.predict(te_d)
        labels = te_l
        err = (sum([1 for p, l in zip(preds, labels)
                    if p != l]) * 1.0 / len(te_l))
        assert err < 0.2

    def test_sklearn_api_gblinear(self):
        self._init_ray()

        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        iris = load_iris()
        tr_d, te_d, tr_l, te_l = train_test_split(
            iris.data, iris.target, train_size=120)

        classifier = RayXGBClassifier(
            booster="gblinear", n_estimators=100, random_state=self.seed)
        classifier.fit(tr_d, tr_l)

        preds = classifier.predict(te_d)
        labels = te_l
        err = (sum([1 for p, l in zip(preds, labels)
                    if p != l]) * 1.0 / len(te_l))
        assert err < 0.5

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 0, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_sklearn_random_state(self):
        self._init_ray()

        clf = RayXGBClassifier(random_state=402)
        assert clf.get_xgb_params()["random_state"] == 402

        clf = RayXGBClassifier(random_state=401)
        assert clf.get_xgb_params()["random_state"] == 401

        random_state = np.random.RandomState(seed=403)
        clf = RayXGBClassifier(random_state=random_state)
        assert isinstance(clf.get_xgb_params()["random_state"], int)

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 0, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_sklearn_n_jobs(self):
        self._init_ray()

        clf = RayXGBClassifier(n_jobs=1)
        assert clf.get_xgb_params()["n_jobs"] == 1

        clf = RayXGBClassifier(n_jobs=2)
        assert clf.get_xgb_params()["n_jobs"] == 2

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 3, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_parameters_access(self):
        self._init_ray()

        from sklearn import datasets

        params = {"updater": "grow_gpu_hist", "subsample": 0.5, "n_jobs": -1}
        clf = RayXGBClassifier(n_estimators=1000, **params)
        assert clf.get_params()["updater"] == "grow_gpu_hist"
        assert clf.get_params()["subsample"] == 0.5
        assert clf.get_params()["n_estimators"] == 1000

        clf = RayXGBClassifier(n_estimators=1, nthread=4)
        X, y = datasets.load_iris(return_X_y=True)
        clf.fit(X, y)

        config = json.loads(clf.get_booster().save_config())
        assert int(config["learner"]["generic_param"]["nthread"]) == 4

        clf.set_params(nthread=16)
        config = json.loads(clf.get_booster().save_config())
        assert int(config["learner"]["generic_param"]["nthread"]) == 16

        clf.predict(X)
        config = json.loads(clf.get_booster().save_config())
        assert int(config["learner"]["generic_param"]["nthread"]) == 16

    def test_kwargs_error(self):
        self._init_ray()

        params = {"updater": "grow_gpu_hist", "subsample": 0.5, "n_jobs": -1}
        with self.assertRaises(TypeError):
            clf = RayXGBClassifier(n_jobs=1000, **params)
            assert isinstance(clf, RayXGBClassifier)

    def test_kwargs_grid_search(self):
        self._init_ray()

        from sklearn.model_selection import GridSearchCV
        from sklearn import datasets

        params = {"tree_method": "hist"}
        clf = RayXGBClassifier(n_estimators=1, learning_rate=1.0, **params)
        assert clf.get_params()["tree_method"] == "hist"
        # 'max_leaves' is not a default argument of XGBClassifier
        # Check we can still do grid search over this parameter
        search_params = {"max_leaves": range(2, 5)}
        grid_cv = GridSearchCV(clf, search_params, cv=5)
        iris = datasets.load_iris()
        grid_cv.fit(iris.data, iris.target)

        # Expect unique results for each parameter value
        # This confirms sklearn is able to successfully update the parameter
        means = grid_cv.cv_results_["mean_test_score"]
        assert len(means) == len(set(means))

    def test_sklearn_clone(self):
        self._init_ray()

        from sklearn.base import clone

        clf = RayXGBClassifier(n_jobs=2)
        clf.n_jobs = -1
        clone(clf)

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 0, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_sklearn_get_default_params(self):
        self._init_ray()

        from sklearn.datasets import load_digits

        digits_2class = load_digits(n_class=2)
        X = digits_2class["data"]
        y = digits_2class["target"]
        cls = RayXGBClassifier()
        assert cls.get_params()["base_score"] is None
        cls.fit(X[:4, ...], y[:4, ...])
        assert cls.get_params()["base_score"] is not None

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 1, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_validation_weights_xgbmodel(self):
        self._init_ray()

        from sklearn.datasets import make_hastie_10_2

        # prepare training and test data
        X, y = make_hastie_10_2(n_samples=2000, random_state=42)
        labels, y = np.unique(y, return_inverse=True)
        X_train, X_test = X[:1600], X[1600:]
        y_train, y_test = y[:1600], y[1600:]

        # instantiate model
        param_dist = {
            "objective": "binary:logistic",
            "n_estimators": 2,
            "random_state": 123,
        }
        clf = xgb.sklearn.XGBModel(**param_dist)

        # train it using instance weights only in the training set
        weights_train = np.random.choice([1, 2], len(X_train))
        clf.fit(
            X_train,
            y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            eval_metric="logloss",
            verbose=False,
        )

        # evaluate logloss metric on test set *without* using weights
        evals_result_without_weights = clf.evals_result()
        logloss_without_weights = evals_result_without_weights["validation_0"][
            "logloss"]

        # now use weights for the test set
        np.random.seed(0)
        weights_test = np.random.choice([1, 2], len(X_test))
        clf.fit(
            X_train,
            y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            sample_weight_eval_set=[weights_test],
            eval_metric="logloss",
            verbose=False,
        )
        evals_result_with_weights = clf.evals_result()
        logloss_with_weights = evals_result_with_weights["validation_0"][
            "logloss"]

        # check that the logloss in the test set is actually different
        # when using weights than when not using them
        assert all((logloss_with_weights[i] != logloss_without_weights[i]
                    for i in [0, 1]))

        with self.assertRaises((ValueError, AssertionError)):
            # length of eval set and sample weight doesn't match.
            clf.fit(
                X_train,
                y_train,
                sample_weight=weights_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                sample_weight_eval_set=[weights_train],
            )

        with self.assertRaises((ValueError, AssertionError)):
            cls = RayXGBClassifier()
            cls.fit(
                X_train,
                y_train,
                sample_weight=weights_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                sample_weight_eval_set=[weights_train],
            )

    def test_validation_weights_xgbclassifier(self):
        self._init_ray()

        from sklearn.datasets import make_hastie_10_2

        # prepare training and test data
        X, y = make_hastie_10_2(n_samples=2000, random_state=42)
        labels, y = np.unique(y, return_inverse=True)
        X_train, X_test = X[:1600], X[1600:]
        y_train, y_test = y[:1600], y[1600:]

        # instantiate model
        param_dist = {
            "objective": "binary:logistic",
            "n_estimators": 2,
            "random_state": 123,
        }
        clf = RayXGBClassifier(**param_dist)

        # train it using instance weights only in the training set
        weights_train = np.random.choice([1, 2], len(X_train))
        clf.fit(
            X_train,
            y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            eval_metric="logloss",
            verbose=False,
        )

        # evaluate logloss metric on test set *without* using weights
        evals_result_without_weights = clf.evals_result()
        logloss_without_weights = evals_result_without_weights["validation_0"][
            "logloss"]

        # now use weights for the test set
        np.random.seed(0)
        weights_test = np.random.choice([1, 2], len(X_test))
        clf.fit(
            X_train,
            y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            sample_weight_eval_set=[weights_test],
            eval_metric="logloss",
            verbose=False,
        )
        evals_result_with_weights = clf.evals_result()
        logloss_with_weights = evals_result_with_weights["validation_0"][
            "logloss"]

        # check that the logloss in the test set is actually different
        # when using weights than when not using them
        assert all((logloss_with_weights[i] != logloss_without_weights[i]
                    for i in [0, 1]))

    def save_load_model(self, model_path):
        from sklearn.datasets import load_digits
        from sklearn.model_selection import KFold

        digits = load_digits(n_class=2)
        y = digits["target"]
        X = digits["data"]
        kf = KFold(n_splits=2, shuffle=True, random_state=self.rng)
        for train_index, test_index in kf.split(X, y):
            xgb_model = RayXGBClassifier(use_label_encoder=False).fit(
                X[train_index], y[train_index])
            xgb_model.save_model(model_path)

            xgb_model = RayXGBClassifier()
            xgb_model.load_model(model_path)

            assert xgb_model.use_label_encoder is False
            assert isinstance(xgb_model.classes_, np.ndarray)
            assert isinstance(xgb_model._Booster, xgb.Booster)

            preds = xgb_model.predict(X[test_index])
            labels = y[test_index]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1
            assert xgb_model.get_booster().attr("scikit_learn") is None

            # test native booster
            preds = xgb_model.predict(X[test_index], output_margin=True)
            booster = xgb.Booster(model_file=model_path)
            predt_1 = booster.predict(
                xgb.DMatrix(X[test_index]), output_margin=True)
            assert np.allclose(preds, predt_1)

            with self.assertRaises(TypeError):
                xgb_model = xgb.XGBModel()
                xgb_model.load_model(model_path)

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 3, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_save_load_model(self):
        self._init_ray()

        with TemporaryDirectory() as tempdir:
            model_path = os.path.join(tempdir, "digits.model")
            self.save_load_model(model_path)

        with TemporaryDirectory() as tempdir:
            model_path = os.path.join(tempdir, "digits.model.json")
            self.save_load_model(model_path)

        from sklearn.datasets import load_digits

        with TemporaryDirectory() as tempdir:
            model_path = os.path.join(tempdir, "digits.model.json")
            digits = load_digits(n_class=2)
            y = digits["target"]
            X = digits["data"]
            booster = xgb.train(
                {
                    "tree_method": "hist",
                    "objective": "binary:logistic"
                },
                dtrain=xgb.DMatrix(X, y),
                num_boost_round=4,
            )
            predt_0 = booster.predict(xgb.DMatrix(X))
            booster.save_model(model_path)
            cls = RayXGBClassifier()
            cls.load_model(model_path)

            proba = cls.predict_proba(X)
            assert proba.shape[0] == X.shape[0]
            assert proba.shape[1] == 2  # binary

            predt_1 = cls.predict_proba(X)[:, 1]
            assert np.allclose(predt_0, predt_1)

            cls = xgb.XGBModel()
            cls.load_model(model_path)
            predt_1 = cls.predict(X)
            assert np.allclose(predt_0, predt_1)

    # # forcing it to be last as it's the longest test by far
    # def test_zzzzzzz_RFECV(self):
    #     self._init_ray()

    #     from sklearn.datasets import load_boston
    #     from sklearn.datasets import load_breast_cancer
    #     from sklearn.datasets import load_iris
    #     from sklearn.feature_selection import RFECV

    #     # Regression
    #     X, y = load_boston(return_X_y=True)
    #     bst = RayXGBRegressor(
    #         booster="gblinear",
    #         learning_rate=0.1,
    #         n_estimators=10,
    #         objective="reg:squarederror",
    #         random_state=0,
    #         verbosity=0,
    #     )
    #     rfecv = RFECV(
    #         estimator=bst, step=1, cv=3, scoring="neg_mean_squared_error")
    #     rfecv.fit(X, y)

    #     # Binary classification
    #     X, y = load_breast_cancer(return_X_y=True)
    #     bst = RayXGBClassifier(
    #         booster="gblinear",
    #         learning_rate=0.1,
    #         n_estimators=10,
    #         objective="binary:logistic",
    #         random_state=0,
    #         verbosity=0,
    #         use_label_encoder=False,
    #     )
    #     rfecv = RFECV(estimator=bst, step=1, cv=3, scoring="roc_auc")
    #     rfecv.fit(X, y)

    #     # Multi-class classification
    #     X, y = load_iris(return_X_y=True)
    #     bst = RayXGBClassifier(
    #         base_score=0.4,
    #         booster="gblinear",
    #         learning_rate=0.1,
    #         n_estimators=10,
    #         objective="multi:softprob",
    #         random_state=0,
    #         reg_alpha=0.001,
    #         reg_lambda=0.01,
    #         scale_pos_weight=0.5,
    #         verbosity=0,
    #         use_label_encoder=False,
    #     )
    #     rfecv = RFECV(estimator=bst, step=1, cv=3, scoring="neg_log_loss")
    #     rfecv.fit(X, y)

    #     X[0:4, :] = np.nan  # verify scikit_learn doesn't throw with nan
    #     reg = RayXGBRegressor()
    #     rfecv = RFECV(estimator=reg)
    #     rfecv.fit(X, y)

    #     cls = RayXGBClassifier(use_label_encoder=False)
    #     rfecv = RFECV(
    #         estimator=cls, step=1, cv=3, scoring="neg_mean_squared_error")
    #     rfecv.fit(X, y)

    def test_XGBClassifier_resume(self):
        self._init_ray()

        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import log_loss

        with TemporaryDirectory() as tempdir:
            model1_path = os.path.join(tempdir, "test_XGBClassifier.model")
            model1_booster_path = os.path.join(tempdir,
                                               "test_XGBClassifier.booster")

            X, Y = load_breast_cancer(return_X_y=True)

            model1 = RayXGBClassifier(
                learning_rate=0.3, random_state=0, n_estimators=8)
            model1.fit(X, Y)

            pred1 = model1.predict(X)
            log_loss1 = log_loss(pred1, Y)

            # file name of stored xgb model
            model1.save_model(model1_path)
            model2 = RayXGBClassifier(
                learning_rate=0.3, random_state=0, n_estimators=8)
            model2.fit(X, Y, xgb_model=model1_path)

            pred2 = model2.predict(X)
            log_loss2 = log_loss(pred2, Y)

            assert np.any(pred1 != pred2)
            assert log_loss1 > log_loss2

            # file name of 'Booster' instance Xgb model
            model1.get_booster().save_model(model1_booster_path)
            model2 = RayXGBClassifier(
                learning_rate=0.3, random_state=0, n_estimators=8)
            model2.fit(X, Y, xgb_model=model1_booster_path)

            pred2 = model2.predict(X)
            log_loss2 = log_loss(pred2, Y)

            assert np.any(pred1 != pred2)
            assert log_loss1 > log_loss2

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 0, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_constraint_parameters(self):
        self._init_ray()

        reg = RayXGBRegressor(interaction_constraints="[[0, 1], [2, 3, 4]]")
        X = np.random.randn(10, 10)
        y = np.random.randn(10)
        reg.fit(X, y)

        config = json.loads(reg.get_booster().save_config())
        assert (config["learner"]["gradient_booster"]["updater"]["prune"][
            "train_param"]["interaction_constraints"] == "[[0, 1], [2, 3, 4]]")

    # TODO check why this is not working (output is empty, probably due to Ray)
    # def test_parameter_validation(self):
    #     self._init_ray()

    #     reg = RayXGBRegressor(foo='bar', verbosity=1)
    #     X = np.random.randn(10, 10)
    #     y = np.random.randn(10)
    #     out = io.StringIO()
    #     err = io.StringIO()
    #     with redirect_stdout(out), redirect_stderr(err):
    #         reg.fit(X, y)
    #     output = out.getvalue().strip()

    #     print(output)
    #     assert output.find('foo') != -1

    #     reg = RayXGBRegressor(n_estimators=2,
    #                           missing=3,
    #                           importance_type='gain',
    #                           verbosity=1)
    #     X = np.random.randn(10, 10)
    #     y = np.random.randn(10)
    #     out = io.StringIO()
    #     err = io.StringIO()
    #     with redirect_stdout(out), redirect_stderr(err):
    #         reg.fit(X, y)
    #     output = out.getvalue().strip()

    #     assert len(output) == 0

    # def test_deprecate_position_arg(self):
    #     self._init_ray()

    #     from sklearn.datasets import load_digits

    #     X, y = load_digits(return_X_y=True, n_class=2)
    #     w = y
    #     with self.assertWarns(FutureWarning):
    #         RayXGBRegressor(3, learning_rate=0.1)
    #     model = RayXGBRegressor(n_estimators=1)
    #     with self.assertWarns(FutureWarning):
    #         model.fit(X, y, w)

    #     with self.assertWarns(FutureWarning):
    #         RayXGBClassifier(1, use_label_encoder=False)
    #     model = RayXGBClassifier(n_estimators=1, use_label_encoder=False)
    #     with self.assertWarns(FutureWarning):
    #         model.fit(X, y, w)

    #     with self.assertWarns(FutureWarning):
    #         RayXGBRanker("rank:ndcg", learning_rate=0.1)
    #     model = RayXGBRanker(n_estimators=1)
    #     group = np.repeat(1, X.shape[0])
    #     with self.assertWarns(FutureWarning):
    #         model.fit(X, y, group)

    #     with self.assertWarns(FutureWarning):
    #         RayXGBRFRegressor(1, learning_rate=0.1)
    #     model = RayXGBRFRegressor(n_estimators=1)
    #     with self.assertWarns(FutureWarning):
    #         model.fit(X, y, w)

    #     with self.assertWarns(FutureWarning):
    #         RayXGBRFClassifier(1, use_label_encoder=True)
    #     model = RayXGBRFClassifier(n_estimators=1)
    #     with self.assertWarns(FutureWarning):
    #         model.fit(X, y, w)

    def test_pandas_input(self):
        self._init_ray()

        import pandas as pd
        from sklearn.calibration import CalibratedClassifierCV

        rng = np.random.RandomState(self.seed)

        kRows = 100
        kCols = 6

        X = rng.randint(low=0, high=2, size=kRows * kCols)
        X = X.reshape(kRows, kCols)

        df = pd.DataFrame(X)
        feature_names = []
        for i in range(1, kCols):
            feature_names += ["k" + str(i)]

        df.columns = ["status"] + feature_names

        target = df["status"]
        train = df.drop(columns=["status"])
        model = RayXGBClassifier()
        model.fit(train, target)
        clf_isotonic = CalibratedClassifierCV(
            model, cv="prefit", method="isotonic")
        clf_isotonic.fit(train, target)
        assert isinstance(
            clf_isotonic.calibrated_classifiers_[0].base_estimator,
            RayXGBClassifier,
        )
        self.assertTrue(
            np.allclose(np.array(clf_isotonic.classes_), np.array([0, 1])))

    # def run_feature_weights(self, X, y, fw, model=RayXGBRegressor):
    #     with TemporaryDirectory() as tmpdir:
    #         colsample_bynode = 0.5
    #         reg = model(
    #           tree_method='hist', colsample_bynode=colsample_bynode
    #         )

    #         reg.fit(X, y, feature_weights=fw)
    #         model_path = os.path.join(tmpdir, 'model.json')
    #         reg.save_model(model_path)
    #         with open(model_path) as fd:
    #             model = json.load(fd)

    #         parser_path = os.path.join(tm.PROJECT_ROOT, 'demo', 'json-model',
    #                                    'json_parser.py')
    #         spec = importlib.util.spec_from_file_location(
    #             "JsonParser", parser_path)
    #         foo = importlib.util.module_from_spec(spec)
    #         spec.loader.exec_module(foo)
    #         model = foo.Model(model)
    #         splits = {}
    #         total_nodes = 0
    #         for tree in model.trees:
    #             n_nodes = len(tree.nodes)
    #             total_nodes += n_nodes
    #             for n in range(n_nodes):
    #                 if tree.is_leaf(n):
    #                     continue
    #                 if splits.get(tree.split_index(n), None) is None:
    #                     splits[tree.split_index(n)] = 1
    #                 else:
    #                     splits[tree.split_index(n)] += 1

    #         od = collections.OrderedDict(sorted(splits.items()))
    #         tuples = [(k, v) for k, v in od.items()]
    #         k, v = list(zip(*tuples))
    #         w = np.polyfit(k, v, deg=1)
    #         return w

    # def test_feature_weights(self):
    #     kRows = 512
    #     kCols = 64
    #     X = self.rng.randn(kRows, kCols)
    #     y = self.rng.randn(kRows)

    #     fw = np.ones(shape=(kCols, ))
    #     for i in range(kCols):
    #         fw[i] *= float(i)
    #     poly_increasing = self.run_feature_weights(X, y, fw, RayXGBRegressor)

    #     fw = np.ones(shape=(kCols, ))
    #     for i in range(kCols):
    #         fw[i] *= float(kCols - i)
    #     poly_decreasing = self.run_feature_weights(X, y, fw, RayXGBRegressor)

    #     # Approxmated test, this is dependent on the implementation of random
    #     # number generator in std library.
    #     assert poly_increasing[0] > 0.08
    #     assert poly_decreasing[0] < -0.08

    def run_boost_from_prediction(self, tree_method):
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        model_0 = RayXGBClassifier(
            learning_rate=0.3,
            random_state=0,
            n_estimators=4,
            tree_method=tree_method,
        )
        model_0.fit(X=X, y=y)
        margin = model_0.predict(X, output_margin=True)

        model_1 = RayXGBClassifier(
            learning_rate=0.3,
            random_state=0,
            n_estimators=4,
            tree_method=tree_method,
        )
        model_1.fit(X=X, y=y, base_margin=margin)
        predictions_1 = model_1.predict(X, base_margin=margin)

        cls_2 = RayXGBClassifier(
            learning_rate=0.3,
            random_state=0,
            n_estimators=8,
            tree_method=tree_method,
        )
        cls_2.fit(X=X, y=y)
        predictions_2 = cls_2.predict(X)
        assert np.all(predictions_1 == predictions_2)

    def boost_from_prediction(self, tree_method):
        self._init_ray()

        self.run_boost_from_prediction(tree_method)

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 0, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_boost_from_prediction_hist(self):
        self.run_boost_from_prediction("hist")

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 2, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_boost_from_prediction_approx(self):
        self.run_boost_from_prediction("approx")

    # Updater `grow_colmaker` or `exact` tree method doesn't support
    # distributed training
    def test_boost_from_prediction_exact(self):
        with self.assertRaises(ValueError):
            self.run_boost_from_prediction("exact")

    @unittest.skipIf(XGBOOST_VERSION_TUPLE < (1, 4, 0),
                     f"not supported in xgb version {xgb.__version__}")
    def test_estimator_type(self):
        self._init_ray()

        assert RayXGBClassifier._estimator_type == "classifier"
        assert RayXGBRFClassifier._estimator_type == "classifier"
        assert RayXGBRegressor._estimator_type == "regressor"
        assert RayXGBRFRegressor._estimator_type == "regressor"
        assert RayXGBRanker._estimator_type == "ranker"

        from sklearn.datasets import load_digits

        X, y = load_digits(n_class=2, return_X_y=True)
        cls = RayXGBClassifier(n_estimators=2).fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cls.json")
            cls.save_model(path)

            reg = RayXGBRegressor()
            with self.assertRaises(TypeError):
                reg.load_model(path)

            cls = RayXGBClassifier()
            cls.load_model(path)  # no error


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
