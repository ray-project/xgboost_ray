"""scikit-learn wrapper for xgboost-ray. Based on xgboost 1.4.0
sklearn wrapper, with some provisions made for 1.5.0 and compatibility
for legacy versions (not everything may work).
Seems to error out with 1.0.0, but 0.90 and 1.1.0 work fine.
Requires xgboost>=0.90"""

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
# https://github.com/dmlc/xgboost/blob/c6a0bdbb5a68232cd59ea556c981c633cc0646ca/python-package/xgboost/sklearn.py

# License:
# https://github.com/dmlc/xgboost/blob/c6a0bdbb5a68232cd59ea556c981c633cc0646ca/LICENSE

from typing import Callable, Tuple, Dict, Optional, Union, Any, List

import numpy as np

import warnings
import functools
import inspect

from xgboost_ray.main import (RayParams, train, predict, XGBOOST_VERSION_TUPLE,
                              LEGACY_WARNING)
from xgboost_ray.matrix import RayDMatrix

from xgboost import Booster, __version__ as xgboost_version
from xgboost.sklearn import (XGBModel, XGBClassifier, XGBRegressor,
                             XGBRFClassifier, XGBRFRegressor, XGBRanker,
                             _objective_decorator)

# avoiding exception in xgboost==0.9.0
try:
    from xgboost.sklearn import _deprecate_positional_args
except ImportError:

    def _deprecate_positional_args(f):
        """Dummy decorator, does nothing"""

        @functools.wraps(f)
        def inner_f(*args, **kwargs):
            return f(*args, **kwargs)

        return inner_f


try:
    from xgboost.sklearn import _wrap_evaluation_matrices
except ImportError:
    # copied from the file in the top comment
    def _wrap_evaluation_matrices(
            missing: float,
            X: Any,
            y: Any,
            group: Optional[Any],
            qid: Optional[Any],
            sample_weight: Optional[Any],
            base_margin: Optional[Any],
            feature_weights: Optional[Any],
            eval_set: Optional[List[Tuple[Any, Any]]],
            sample_weight_eval_set: Optional[List[Any]],
            base_margin_eval_set: Optional[List[Any]],
            eval_group: Optional[List[Any]],
            eval_qid: Optional[List[Any]],
            create_dmatrix: Callable,
            label_transform: Callable = lambda x: x,
    ) -> Tuple[Any, Optional[List[Tuple[Any, str]]]]:
        """Convert array_like evaluation matrices into DMatrix.
        Perform validation on the way.
        """
        train_dmatrix = create_dmatrix(
            data=X,
            label=label_transform(y),
            group=group,
            qid=qid,
            weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            missing=missing,
        )

        n_validation = 0 if eval_set is None else len(eval_set)

        def validate_or_none(meta: Optional[List], name: str) -> List:
            if meta is None:
                return [None] * n_validation
            if len(meta) != n_validation:
                raise ValueError(
                    f"{name}'s length does not eqaul to `eval_set`, " +
                    f"expecting {n_validation}, got {len(meta)}")
            return meta

        if eval_set is not None:
            sample_weight_eval_set = validate_or_none(
                sample_weight_eval_set, "sample_weight_eval_set")
            base_margin_eval_set = validate_or_none(base_margin_eval_set,
                                                    "base_margin_eval_set")
            eval_group = validate_or_none(eval_group, "eval_group")
            eval_qid = validate_or_none(eval_qid, "eval_qid")

            evals = []
            for i, (valid_X, valid_y) in enumerate(eval_set):
                # Skip the duplicated entry.
                if all((valid_X is X, valid_y is y,
                        sample_weight_eval_set[i] is sample_weight,
                        base_margin_eval_set[i] is base_margin,
                        eval_group[i] is group, eval_qid[i] is qid)):
                    evals.append(train_dmatrix)
                else:
                    m = create_dmatrix(
                        data=valid_X,
                        label=label_transform(valid_y),
                        weight=sample_weight_eval_set[i],
                        group=eval_group[i],
                        qid=eval_qid[i],
                        base_margin=base_margin_eval_set[i],
                        missing=missing,
                    )
                    evals.append(m)
            nevals = len(evals)
            eval_names = ["validation_{}".format(i) for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            if any(meta is not None for meta in [
                    sample_weight_eval_set,
                    base_margin_eval_set,
                    eval_group,
                    eval_qid,
            ]):
                raise ValueError(
                    "`eval_set` is not set but one of the other evaluation "
                    "meta info is not None.")
            evals = []

        return train_dmatrix, evals


try:
    from xgboost.sklearn import _convert_ntree_limit
except ImportError:
    _convert_ntree_limit = None

try:
    from xgboost.sklearn import _cls_predict_proba
except ImportError:
    # copied from the file in the top comment
    def _cls_predict_proba(n_classes: int, prediction, vstack: Callable):
        assert len(prediction.shape) <= 2
        if len(prediction.shape) == 2 and prediction.shape[1] == n_classes:
            return prediction
        # binary logistic function
        classone_probs = prediction
        classzero_probs = 1.0 - classone_probs
        return vstack((classzero_probs, classone_probs)).transpose()


try:
    from xgboost.sklearn import _is_cudf_df, _is_cudf_ser, _is_cupy_array
except ImportError:
    _is_cudf_df = None
    _is_cudf_ser = None
    _is_cupy_array = None

try:
    from xgboost.compat import XGBoostLabelEncoder
except ImportError:
    from sklearn.preprocessing import LabelEncoder as XGBoostLabelEncoder

_RAY_PARAMS_DOC = """ray_params (Union[None, RayParams, Dict]): Parameters to configure
            Ray-specific behavior. See :class:`RayParams` for a list of valid
            configuration parameters. Will override ``n_jobs`` attribute
            with own ``num_actors`` parameter.
        _remote (bool): Whether to run the driver process in a remote
            function. This is enabled by default in Ray client mode.
        ray_dmatrix_params (Optional[Dict]): Dict of parameters
            (such as sharding mode) passed to the internal RayDMatrix
            initialization.

"""

_N_JOBS_DOC_REPLACE = (
    """    n_jobs : int
        Number of parallel threads used to run xgboost.  When used with other Scikit-Learn
        algorithms like grid search, you may choose which algorithm to parallelize and
        balance the threads.  Creating thread contention will significantly slow down both
        algorithms.""",  # noqa: E501, W291
    """    n_jobs : int
        Number of Ray actors used to run xgboost in parallel.
        In order to set number of threads per actor, pass a :class:`RayParams`
        object to the relevant method as a ``ray_params`` argument. Will be
        overriden by the ``num_actors`` parameter of ``ray_params`` argument
        should it be passed to a method.""",  # noqa: E501, W291
)


def _treat_estimator_doc(doc: str) -> str:
    """Helper function to make nececssary changes in estimator docstrings"""
    doc = doc.replace(*_N_JOBS_DOC_REPLACE).replace(
        "scikit-learn API for XGBoost",
        "scikit-learn API for Ray-distributed XGBoost")
    return doc


def _treat_X_doc(doc: str) -> str:
    doc = doc.replace("Data to predict with.",
                      "Data to predict with. Can also be a ``RayDMatrix``.")
    doc = doc.replace("Feature matrix.",
                      "Feature matrix. Can also be a ``RayDMatrix``.")
    doc = doc.replace("Feature matrix",
                      "Feature matrix. Can also be a ``RayDMatrix``.")
    return doc


def _xgboost_version_warn(f):
    """Decorator to warn when xgboost version is < 1.4.0"""

    @functools.wraps(f)
    def inner_f(*args, **kwargs):
        if XGBOOST_VERSION_TUPLE < (1, 4, 0):
            warnings.warn(LEGACY_WARNING)
        return f(*args, **kwargs)

    return inner_f


def _check_if_params_are_ray_dmatrix(X, sample_weight, base_margin, eval_set,
                                     sample_weight_eval_set,
                                     base_margin_eval_set):
    train_dmatrix = None
    evals = ()
    eval_set = eval_set or ()
    if isinstance(X, RayDMatrix):
        params_to_warn_about = ["y"]
        if sample_weight is not None:
            params_to_warn_about.append("sample_weight")
        if base_margin is not None:
            params_to_warn_about.append("base_margin")
        warnings.warn(f"X is a RayDMatrix, {', '.join(params_to_warn_about)}"
                      " will be ignored!")
        train_dmatrix = X
        if eval_set:
            if any(not isinstance(eval_data, RayDMatrix)
                   or not isinstance(eval_name, str)
                   for eval_data, eval_name in eval_set):
                raise ValueError("If X is a RayDMatrix, all elements of "
                                 "`eval_set` must be (RayDMatrix, str) "
                                 "tuples.")
        params_to_warn_about = []
        if sample_weight_eval_set is not None:
            params_to_warn_about.append("sample_weight_eval_set")
        if base_margin_eval_set is not None:
            params_to_warn_about.append("base_margin_eval_set")
        if params_to_warn_about:
            warnings.warn(
                "`eval_set` is composed of RayDMatrix tuples, "
                f"{', '.join(params_to_warn_about)} will be ignored!")
        evals = eval_set or ()
    elif any(
            isinstance(eval_x, RayDMatrix) or isinstance(eval_y, RayDMatrix)
            for eval_x, eval_y in eval_set):
        raise ValueError("If X is not a RayDMatrix, all `eval_set` "
                         "elements must be (array_like, array_like)"
                         " tuples.")
    return train_dmatrix, evals


class RayXGBMixin:
    """Mixin class to provide xgboost-ray functionality"""

    def _ray_set_ray_params_n_jobs(
            self, ray_params: Optional[Union[RayParams, dict]],
            n_jobs: Optional[int]) -> RayParams:
        """Helper function to set num_actors in ray_params if not
        set by the user"""
        if ray_params is None:
            if not n_jobs or n_jobs < 1:
                n_jobs = 1
            ray_params = RayParams(num_actors=n_jobs)
        elif n_jobs is not None:
            warnings.warn("`ray_params` is not `None` and will override "
                          "the `n_jobs` attribute.")
        return ray_params

    def _ray_predict(
            self: "XGBModel",
            X,
            output_margin=False,
            ntree_limit=None,
            validate_features=True,
            base_margin=None,
            iteration_range=None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
    ):
        """Distributed predict via Ray"""
        compat_predict_kwargs = {}
        if _convert_ntree_limit is not None:
            iteration_range = _convert_ntree_limit(
                self.get_booster(), ntree_limit, iteration_range)
            iteration_range = self._get_iteration_range(iteration_range)
            compat_predict_kwargs["iteration_range"] = iteration_range
        else:
            if ntree_limit is None:
                ntree_limit = getattr(self, "best_ntree_limit", 0)
            compat_predict_kwargs["ntree_limit"] = ntree_limit

        ray_params = self._ray_set_ray_params_n_jobs(ray_params, self.n_jobs)
        ray_dmatrix_params = ray_dmatrix_params or {}

        if not isinstance(X, RayDMatrix):
            test = RayDMatrix(
                X,
                base_margin=base_margin,
                missing=self.missing,
                **ray_dmatrix_params)
        else:
            test = X
            if base_margin is not None:
                warnings.warn(
                    "X is a RayDMatrix, base_margin will be ignored!")

        return predict(
            self.get_booster(),
            data=test,
            output_margin=output_margin,
            validate_features=validate_features,
            ray_params=ray_params,
            _remote=_remote,
            **compat_predict_kwargs,
        )

    def _ray_get_wrap_evaluation_matrices_compat_kwargs(self) -> dict:
        if hasattr(self, "enable_categorical"):
            return {"enable_categorical": self.enable_categorical}
        return {}

    # copied from the file in the top comment
    # provided here for compatibility with legacy xgboost versions
    # will be overwritten by vanilla xgboost if possible
    def _configure_fit(
            self,
            booster: Optional[Union[Booster, "XGBModel", str]],
            eval_metric: Optional[Union[Callable, str, List[str]]],
            params: Dict[str, Any],
    ) -> Tuple[Optional[Union[Booster, str]], Dict[str, Any]]:
        # pylint: disable=protected-access, no-self-use
        if isinstance(booster, XGBModel):
            # Handle the case when xgb_model is a sklearn model object
            model: Optional[Union[Booster, str]] = booster._Booster
        else:
            model = booster

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({"eval_metric": eval_metric})
        return model, feval, params

    # copied from the file in the top comment
    # provided here for compatibility with legacy xgboost versions
    # will be overwritten by vanilla xgboost if possible
    def _set_evaluation_result(self, evals_result) -> None:
        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][
                    evals_result_key]
            self.evals_result_ = evals_result


class RayXGBRegressor(XGBRegressor, RayXGBMixin):
    __init__ = _xgboost_version_warn(XGBRegressor.__init__)

    @_deprecate_positional_args
    def fit(
            self,
            X,
            y,
            *,
            sample_weight=None,
            base_margin=None,
            eval_set=None,
            eval_metric=None,
            early_stopping_rounds=None,
            verbose=True,
            xgb_model: Optional[Union[Booster, str, "XGBModel"]] = None,
            sample_weight_eval_set=None,
            base_margin_eval_set=None,
            feature_weights=None,
            callbacks=None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
    ):
        evals_result = {}
        ray_dmatrix_params = ray_dmatrix_params or {}

        train_dmatrix, evals = _check_if_params_are_ray_dmatrix(
            X, sample_weight, base_margin, eval_set, sample_weight_eval_set,
            base_margin_eval_set)

        if train_dmatrix is None:
            train_dmatrix, evals = _wrap_evaluation_matrices(
                missing=self.missing,
                X=X,
                y=y,
                group=None,
                qid=None,
                sample_weight=sample_weight,
                base_margin=base_margin,
                feature_weights=feature_weights,
                eval_set=eval_set,
                sample_weight_eval_set=sample_weight_eval_set,
                base_margin_eval_set=base_margin_eval_set,
                eval_group=None,
                eval_qid=None,
                # changed in xgboost-ray:
                create_dmatrix=lambda **kwargs: RayDMatrix(**{
                    **kwargs,
                    **ray_dmatrix_params
                }),
                **self._ray_get_wrap_evaluation_matrices_compat_kwargs())

        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            params["objective"] = "reg:squarederror"
        else:
            obj = None

        model, feval, params = self._configure_fit(xgb_model, eval_metric,
                                                   params)

        # remove those as they will be set in RayXGBoostActor
        params.pop("n_jobs", None)
        params.pop("nthread", None)

        ray_params = self._ray_set_ray_params_n_jobs(ray_params, self.n_jobs)

        additional_results = {}

        self._Booster = train(
            params,
            train_dmatrix,
            self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            obj=obj,
            feval=feval,
            verbose_eval=verbose,
            xgb_model=model,
            callbacks=callbacks,
            # changed in xgboost-ray:
            additional_results=additional_results,
            ray_params=ray_params,
            _remote=_remote,
        )

        self.additional_results_ = additional_results

        self._set_evaluation_result(evals_result)
        return self

    fit.__doc__ = _treat_X_doc(XGBRegressor.fit.__doc__) + _RAY_PARAMS_DOC

    def _can_use_inplace_predict(self) -> bool:
        return False

    def predict(
            self,
            X,
            output_margin=False,
            ntree_limit=None,
            validate_features=True,
            base_margin=None,
            iteration_range=None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
    ):
        return self._ray_predict(
            X,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params)

    predict.__doc__ = _treat_X_doc(
        XGBRegressor.predict.__doc__) + _RAY_PARAMS_DOC

    def load_model(self, fname):
        if not hasattr(self, "_Booster"):
            self._Booster = Booster()
        return super().load_model(fname)


RayXGBRegressor.__doc__ = _treat_estimator_doc(XGBRegressor.__doc__)


class RayXGBRFRegressor(RayXGBRegressor):

    # too much work to make this compatible with 0.90
    if xgboost_version == "0.90":

        def __init__(self, *args, **kwargs):
            raise ValueError(
                "RayXGBRFRegressor not available with xgboost<1.0.0")
    else:

        @_deprecate_positional_args
        @_xgboost_version_warn
        def __init__(self,
                     *,
                     learning_rate=1,
                     subsample=0.8,
                     colsample_bynode=0.8,
                     reg_lambda=1e-5,
                     **kwargs):
            super().__init__(
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bynode=colsample_bynode,
                reg_lambda=reg_lambda,
                **kwargs)

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self):
        return 1


RayXGBRFRegressor.__doc__ = _treat_estimator_doc(XGBRFRegressor.__doc__)


class RayXGBClassifier(XGBClassifier, RayXGBMixin):
    __init__ = _xgboost_version_warn(XGBClassifier.__init__)

    @_deprecate_positional_args
    def fit(
            self,
            X,
            y,
            *,
            sample_weight=None,
            base_margin=None,
            eval_set=None,
            eval_metric=None,
            early_stopping_rounds=None,
            verbose=True,
            xgb_model=None,
            sample_weight_eval_set=None,
            base_margin_eval_set=None,
            feature_weights=None,
            callbacks=None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
    ):

        evals_result = {}
        ray_dmatrix_params = ray_dmatrix_params or {}

        params = self.get_xgb_params()

        train_dmatrix, evals = _check_if_params_are_ray_dmatrix(
            X, sample_weight, base_margin, eval_set, sample_weight_eval_set,
            base_margin_eval_set)

        if train_dmatrix is not None:
            if not hasattr(self, "use_label_encoder"):
                warnings.warn("If X is a RayDMatrix, no label encoding"
                              " will be performed. Ensure the labels are"
                              " encoded.")
            elif self.use_label_encoder:
                raise ValueError(
                    "X cannot be a RayDMatrix if `use_label_encoder` "
                    "is set to True")
            if "num_class" not in params:
                raise ValueError(
                    "`num_class` must be set during initalization if X"
                    " is a RayDMatrix")
            self.classes_ = list(range(0, params["num_class"]))
            self.n_classes_ = params["num_class"]
            if self.n_classes_ <= 2:
                params.pop("num_class")
            label_transform = lambda x: x  # noqa: E731
        else:
            if len(X.shape) != 2:
                # Simply raise an error here since there might be many
                # different ways of reshaping
                raise ValueError(
                    "Please reshape the input data X into 2-dimensional "
                    "matrix.")

            label_transform = self._ray_fit_preprocess(y)

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            params["objective"] = "binary:logistic"
        else:
            obj = None

        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying
            # XGB instance
            params["objective"] = "multi:softprob"
            params["num_class"] = self.n_classes_

        model, feval, params = self._configure_fit(xgb_model, eval_metric,
                                                   params)

        if train_dmatrix is None:
            train_dmatrix, evals = _wrap_evaluation_matrices(
                missing=self.missing,
                X=X,
                y=y,
                group=None,
                qid=None,
                sample_weight=sample_weight,
                base_margin=base_margin,
                feature_weights=feature_weights,
                eval_set=eval_set,
                sample_weight_eval_set=sample_weight_eval_set,
                base_margin_eval_set=base_margin_eval_set,
                eval_group=None,
                eval_qid=None,
                label_transform=label_transform,
                # changed in xgboost-ray:
                create_dmatrix=lambda **kwargs: RayDMatrix(**{
                    **kwargs,
                    **ray_dmatrix_params
                }),
                **self._ray_get_wrap_evaluation_matrices_compat_kwargs())

        # remove those as they will be set in RayXGBoostActor
        params.pop("n_jobs", None)
        params.pop("nthread", None)

        ray_params = self._ray_set_ray_params_n_jobs(ray_params, self.n_jobs)

        additional_results = {}

        self._Booster = train(
            params,
            train_dmatrix,
            self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            obj=obj,
            feval=feval,
            verbose_eval=verbose,
            xgb_model=model,
            callbacks=callbacks,
            # changed in xgboost-ray:
            additional_results=additional_results,
            ray_params=ray_params,
            _remote=_remote,
        )

        if not callable(self.objective):
            self.objective = params["objective"]

        self.additional_results_ = additional_results

        self._set_evaluation_result(evals_result)
        return self

    fit.__doc__ = _treat_X_doc(XGBClassifier.fit.__doc__) + _RAY_PARAMS_DOC

    def _ray_fit_preprocess(self, y) -> Callable:
        """This has been separated out so that it can be easily overwritten
        should a future xgboost version remove label encoding"""
        # pylint: disable = attribute-defined-outside-init,too-many-statements
        can_use_label_encoder = True
        use_label_encoder = getattr(self, "use_label_encoder", True)
        label_encoding_check_error = (
            "The label must consist of integer "
            "labels of form 0, 1, 2, ..., [num_class - 1].")
        label_encoder_deprecation_msg = (
            "The use of label encoder in XGBClassifier is deprecated and will "
            "be removed in a future release. To remove this warning, do the "
            "following: 1) Pass option use_label_encoder=False when "
            "constructing XGBClassifier object; and 2) Encode your labels (y) "
            "as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].")

        # ray: modified this to allow for compatibility with legacy xgboost
        if (_is_cudf_df and _is_cudf_df(y)) or (_is_cudf_ser
                                                and _is_cudf_ser(y)):
            import cupy as cp  # pylint: disable=E0401

            self.classes_ = cp.unique(y.values)
            self.n_classes_ = len(self.classes_)
            can_use_label_encoder = False
            expected_classes = cp.arange(self.n_classes_)
            if (self.classes_.shape != expected_classes.shape
                    or not (self.classes_ == expected_classes).all()):
                raise ValueError(label_encoding_check_error)
        elif (_is_cupy_array and _is_cupy_array(y)):
            import cupy as cp  # pylint: disable=E0401

            self.classes_ = cp.unique(y)
            self.n_classes_ = len(self.classes_)
            can_use_label_encoder = False
            expected_classes = cp.arange(self.n_classes_)
            if (self.classes_.shape != expected_classes.shape
                    or not (self.classes_ == expected_classes).all()):
                raise ValueError(label_encoding_check_error)
        else:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            if not use_label_encoder and (not np.array_equal(
                    self.classes_, np.arange(self.n_classes_))):
                raise ValueError(label_encoding_check_error)

        if use_label_encoder:
            if not can_use_label_encoder:
                raise ValueError(
                    "The option use_label_encoder=True is incompatible with "
                    "inputs of type cuDF or cuPy. Please set "
                    "use_label_encoder=False when  constructing XGBClassifier "
                    "object. NOTE:" + label_encoder_deprecation_msg)
            if hasattr(self, "use_label_encoder"):
                warnings.warn(label_encoder_deprecation_msg, UserWarning)
            self._le = XGBoostLabelEncoder().fit(y)
            label_transform = self._le.transform
        else:
            label_transform = lambda x: x  # noqa: E731

        return label_transform

    def _can_use_inplace_predict(self) -> bool:
        return False

    def predict(
            self,
            X,
            output_margin=False,
            ntree_limit=None,
            validate_features=True,
            base_margin=None,
            iteration_range: Optional[Tuple[int, int]] = None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
    ):
        class_probs = self._ray_predict(
            X=X,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params)
        if output_margin:
            # If output_margin is active, simply return the scores
            return class_probs

        if len(class_probs.shape) > 1:
            # turns softprob into softmax
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            # turns soft logit into class label
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1

        if hasattr(self, "_le"):
            return self._le.inverse_transform(column_indexes)
        return column_indexes

    predict.__doc__ = _treat_X_doc(XGBModel.predict.__doc__) + _RAY_PARAMS_DOC

    def predict_proba(
            self,
            X,
            ntree_limit=None,
            validate_features=False,
            base_margin=None,
            iteration_range: Optional[Tuple[int, int]] = None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
    ) -> np.ndarray:

        class_probs = self._ray_predict(
            X=X,
            output_margin=self.objective == "multi:softmax",
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params)
        # If model is loaded from a raw booster there's no `n_classes_`
        return _cls_predict_proba(
            getattr(self, "n_classes_", None), class_probs, np.vstack)

    def load_model(self, fname):
        if not hasattr(self, "_Booster"):
            self._Booster = Booster()
        return super().load_model(fname)

    predict_proba.__doc__ = (
        _treat_X_doc(XGBClassifier.predict_proba.__doc__) + _RAY_PARAMS_DOC)


RayXGBClassifier.__doc__ = _treat_estimator_doc(XGBClassifier.__doc__)


class RayXGBRFClassifier(RayXGBClassifier):
    # too much work to make this compatible with 0.90
    if xgboost_version == "0.90":

        def __init__(self, *args, **kwargs):
            raise ValueError(
                "RayXGBRFClassifier not available with xgboost<1.0.0")

    # use_label_encoder added in xgboost commit
    # c8ec62103a36f1717d032b1ddff2bf9e0642508a (1.3.0)
    elif "use_label_encoder" in inspect.signature(
            XGBRFClassifier.__init__).parameters:

        @_deprecate_positional_args
        @_xgboost_version_warn
        def __init__(self,
                     *,
                     learning_rate=1,
                     subsample=0.8,
                     colsample_bynode=0.8,
                     reg_lambda=1e-5,
                     use_label_encoder=True,
                     **kwargs):
            super().__init__(
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bynode=colsample_bynode,
                reg_lambda=reg_lambda,
                use_label_encoder=use_label_encoder,
                **kwargs)
    else:

        @_deprecate_positional_args
        @_xgboost_version_warn
        def __init__(self,
                     *,
                     learning_rate=1,
                     subsample=0.8,
                     colsample_bynode=0.8,
                     reg_lambda=1e-5,
                     **kwargs):
            super().__init__(
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bynode=colsample_bynode,
                reg_lambda=reg_lambda,
                **kwargs)

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self):
        return 1


RayXGBRFClassifier.__doc__ = _treat_estimator_doc(XGBRFClassifier.__doc__)


class RayXGBRanker(XGBRanker, RayXGBMixin):
    __init__ = _xgboost_version_warn(XGBRanker.__init__)

    @_deprecate_positional_args
    def fit(
            self,
            X,
            y,
            *,
            group=None,
            qid=None,
            sample_weight=None,
            base_margin=None,
            eval_set=None,
            eval_group=None,
            eval_qid=None,
            eval_metric=None,
            early_stopping_rounds=None,
            verbose=False,
            xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
            sample_weight_eval_set=None,
            base_margin_eval_set=None,
            feature_weights=None,
            callbacks=None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
    ):

        # check if group information is provided
        if group is None and qid is None:
            raise ValueError("group or qid is required for ranking task")

        if eval_set is not None:
            if eval_group is None and eval_qid is None:
                raise ValueError("eval_group or eval_qid is required if"
                                 " eval_set is not None")

        train_dmatrix, evals = _check_if_params_are_ray_dmatrix(
            X, sample_weight, base_margin, eval_set, sample_weight_eval_set,
            base_margin_eval_set)

        if train_dmatrix is None:
            train_dmatrix, evals = _wrap_evaluation_matrices(
                missing=self.missing,
                X=X,
                y=y,
                group=group,
                qid=qid,
                sample_weight=sample_weight,
                base_margin=base_margin,
                feature_weights=feature_weights,
                eval_set=eval_set,
                sample_weight_eval_set=sample_weight_eval_set,
                base_margin_eval_set=base_margin_eval_set,
                eval_group=eval_group,
                eval_qid=eval_qid,
                # changed in xgboost-ray:
                create_dmatrix=lambda **kwargs: RayDMatrix(**{
                    **kwargs,
                    **ray_dmatrix_params
                }),
                **self._ray_get_wrap_evaluation_matrices_compat_kwargs())

        evals_result = {}
        params = self.get_xgb_params()

        model, feval, params = self._configure_fit(xgb_model, eval_metric,
                                                   params)
        if callable(feval):
            raise ValueError(
                "Custom evaluation metric is not yet supported for XGBRanker.")

        # remove those as they will be set in RayXGBoostActor
        params.pop("n_jobs", None)
        params.pop("nthread", None)

        ray_params = self._ray_set_ray_params_n_jobs(ray_params, self.n_jobs)

        additional_results = {}

        self._Booster = train(
            params,
            train_dmatrix,
            self.n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            evals=evals,
            evals_result=evals_result,
            feval=feval,
            verbose_eval=verbose,
            xgb_model=model,
            callbacks=callbacks,
            # changed in xgboost-ray:
            additional_results=additional_results,
            ray_params=ray_params,
            _remote=_remote,
        )

        self.objective = params["objective"]

        self.additional_results_ = additional_results

        self._set_evaluation_result(evals_result)
        return self

    fit.__doc__ = _treat_X_doc(XGBRanker.fit.__doc__) + _RAY_PARAMS_DOC

    def _can_use_inplace_predict(self) -> bool:
        return False

    def predict(
            self,
            X,
            output_margin=False,
            ntree_limit=None,
            validate_features=True,
            base_margin=None,
            iteration_range=None,
            ray_params: Union[None, RayParams, Dict] = None,
            _remote: Optional[bool] = None,
            ray_dmatrix_params: Optional[Dict] = None,
    ):
        return self._ray_predict(
            X,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
            ray_params=ray_params,
            _remote=_remote,
            ray_dmatrix_params=ray_dmatrix_params)

    predict.__doc__ = _treat_X_doc(XGBRanker.predict.__doc__) + _RAY_PARAMS_DOC

    def load_model(self, fname):
        if not hasattr(self, "_Booster"):
            self._Booster = Booster()
        return super().load_model(fname)


RayXGBRanker.__doc__ = _treat_estimator_doc(XGBRanker.__doc__)
