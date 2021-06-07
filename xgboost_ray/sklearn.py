from typing import Tuple, Dict, Optional, Union

import numpy as np

import warnings
import functools

from xgboost import Booster
from xgboost.sklearn import (XGBModel, XGBClassifier, XGBRegressor,
                             _objective_decorator, _wrap_evaluation_matrices,
                             _convert_ntree_limit, _is_cudf_df, _is_cudf_ser,
                             _is_cupy_array, _cls_predict_proba)

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


from xgboost.compat import XGBoostLabelEncoder

from xgboost_ray.main import RayParams, train, predict
from xgboost_ray.matrix import RayDMatrix

_RAY_PARAMS_DOC = """
    ray_params (Union[None, RayParams, Dict]): Parameters to configure
        Ray-specific behavior. See :class:`RayParams` for a list of valid
        configuration parameters.
    _remote (bool): Whether to run the driver process in a remote
        function. This is enabled by default in Ray client mode.

"""

_N_JOBS_DOC_REPLACE = (
    """    n_jobs : int
        Number of parallel threads used to run xgboost.  When used with other Scikit-Learn
        algorithms like grid search, you may choose which algorithm to parallelize and
        balance the threads.  Creating thread contention will significantly slow down both
        algorithms.""",  # noqa: E501, W291
    """    n_jobs : int
        Number of Ray actors used to run xgboost in parallel.
        In order to set number of threads per actor, pass a ``RayParams`` object to the 
        relevant method as a ``ray_params`` argument.""",  # noqa: E501, W291
)


def _treat_estimator_doc(doc: str) -> str:
    """Helper function to make nececssary changes in estimator docstrings"""
    doc = doc.replace(*_N_JOBS_DOC_REPLACE).replace(
        "Implementation of the scikit-learn API for XGBoost",
        "Implementation of the scikit-learn API for Ray-distributed XGBoost")
    return doc


# would normally use a mixin class but it breaks xgb's get_params
def _predict(
        model: XGBModel,
        X,
        output_margin=False,
        ntree_limit=None,
        validate_features=True,
        base_margin=None,
        iteration_range=None,
        ray_params: Union[None, RayParams, Dict] = None,
        _remote: Optional[bool] = None,
):
    iteration_range = _convert_ntree_limit(model.get_booster(), ntree_limit,
                                           iteration_range)
    iteration_range = model._get_iteration_range(iteration_range)

    if ray_params is None:
        # TODO warning here?
        n_jobs = model.n_jobs
        if not n_jobs or n_jobs < 1:
            n_jobs = 1
        ray_params = RayParams(num_actors=n_jobs)

    test = RayDMatrix(X, base_margin=base_margin, missing=model.missing)
    return predict(
        model.get_booster(),
        data=test,
        iteration_range=iteration_range,
        output_margin=output_margin,
        validate_features=validate_features,
        ray_params=ray_params,
        _remote=_remote,
    )


class RayXGBRegressor(XGBRegressor):
    @_deprecate_positional_args
    def fit(self,
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
            _remote: Optional[bool] = None):
        evals_result = {}

        # enable_categorical param has been added in xgboost 1.5.0
        try:
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
                create_dmatrix=lambda **kwargs: RayDMatrix(**kwargs),
                enable_categorical=self.enable_categorical,
            )
        except AttributeError as e:
            if "enable_categorical" not in str(e):
                raise e
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
                create_dmatrix=lambda **kwargs: RayDMatrix(**kwargs),
            )

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

        if ray_params is None:
            # TODO warning here?
            n_jobs = self.n_jobs
            if not n_jobs or n_jobs < 1:
                n_jobs = 1
            ray_params = RayParams(num_actors=n_jobs)

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
            ray_params=ray_params,
            _remote=_remote,
        )

        self._set_evaluation_result(evals_result)
        return self

    fit.__doc__ = XGBRegressor.fit.__doc__ + _RAY_PARAMS_DOC

    def _can_use_inplace_predict(self) -> bool:
        return False

    def predict(self,
                X,
                output_margin=False,
                ntree_limit=None,
                validate_features=True,
                base_margin=None,
                iteration_range=None,
                ray_params: Union[None, RayParams, Dict] = None,
                _remote: Optional[bool] = None):
        return _predict(
            self,
            X,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
            ray_params=ray_params,
            _remote=_remote)

    predict.__doc__ = XGBRegressor.predict.__doc__ + _RAY_PARAMS_DOC

    def load_model(self, fname):
        if not hasattr(self, "_Booster"):
            self._Booster = Booster()
        return super().load_model(fname)


RayXGBRegressor.__doc__ = _treat_estimator_doc(XGBRegressor.__doc__)


class RayXGBClassifier(XGBClassifier):
    @_deprecate_positional_args
    def fit(self,
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
            _remote: Optional[bool] = None):
        # pylint: disable = attribute-defined-outside-init,too-many-statements
        can_use_label_encoder = True
        label_encoding_check_error = (
            "The label must consist of integer "
            "labels of form 0, 1, 2, ..., [num_class - 1].")
        label_encoder_deprecation_msg = (
            "The use of label encoder in XGBClassifier is deprecated and will "
            "be removed in a future release. To remove this warning, do the "
            "following: 1) Pass option use_label_encoder=False when "
            "constructing XGBClassifier object; and 2) Encode your labels (y) "
            "as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].")

        evals_result = {}
        if _is_cudf_df(y) or _is_cudf_ser(y):
            import cupy as cp  # pylint: disable=E0401

            self.classes_ = cp.unique(y.values)
            self.n_classes_ = len(self.classes_)
            can_use_label_encoder = False
            expected_classes = cp.arange(self.n_classes_)
            if (self.classes_.shape != expected_classes.shape
                    or not (self.classes_ == expected_classes).all()):
                raise ValueError(label_encoding_check_error)
        elif _is_cupy_array(y):
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
            if not self.use_label_encoder and (not np.array_equal(
                    self.classes_, np.arange(self.n_classes_))):
                raise ValueError(label_encoding_check_error)

        params = self.get_xgb_params()

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

        if self.use_label_encoder:
            if not can_use_label_encoder:
                raise ValueError(
                    "The option use_label_encoder=True is incompatible with "
                    "inputs of type cuDF or cuPy. Please set "
                    "use_label_encoder=False when  constructing XGBClassifier "
                    "object. NOTE:" + label_encoder_deprecation_msg)
            warnings.warn(label_encoder_deprecation_msg, UserWarning)
            self._le = XGBoostLabelEncoder().fit(y)
            label_transform = self._le.transform
        else:
            label_transform = lambda x: x  # noqa: E731

        model, feval, params = self._configure_fit(xgb_model, eval_metric,
                                                   params)
        if len(X.shape) != 2:
            # Simply raise an error here since there might be many
            # different ways of reshaping
            raise ValueError(
                "Please reshape the input data X into 2-dimensional matrix.")

        # enable_categorical param has been added in xgboost 1.5.0
        try:
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
                create_dmatrix=lambda **kwargs: RayDMatrix(**kwargs),
                label_transform=label_transform,
                enable_categorical=self.enable_categorical,
            )
        except AttributeError as e:
            if "enable_categorical" not in str(e):
                raise e
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
                create_dmatrix=lambda **kwargs: RayDMatrix(**kwargs),
                label_transform=label_transform,
            )

        # remove those as they will be set in RayXGBoostActor
        params.pop("n_jobs", None)
        params.pop("nthread", None)

        if ray_params is None:
            # TODO warning here?
            n_jobs = self.n_jobs
            if not n_jobs or n_jobs < 1:
                n_jobs = 1
            ray_params = RayParams(num_actors=n_jobs)

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
            ray_params=ray_params,
            _remote=_remote,
        )

        if not callable(self.objective):
            self.objective = params["objective"]

        self._set_evaluation_result(evals_result)
        return self

    fit.__doc__ = XGBClassifier.fit.__doc__ + _RAY_PARAMS_DOC

    def _can_use_inplace_predict(self) -> bool:
        return False

    def predict(self,
                X,
                output_margin=False,
                ntree_limit=None,
                validate_features=True,
                base_margin=None,
                iteration_range: Optional[Tuple[int, int]] = None,
                ray_params: Union[None, RayParams, Dict] = None,
                _remote: Optional[bool] = None):
        class_probs = _predict(
            self,
            X=X,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
            ray_params=ray_params,
            _remote=_remote,
        )
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

    predict.__doc__ = XGBModel.predict.__doc__ + _RAY_PARAMS_DOC

    def predict_proba(self,
                      X,
                      ntree_limit=None,
                      validate_features=False,
                      base_margin=None,
                      iteration_range: Optional[Tuple[int, int]] = None,
                      ray_params: Union[None, RayParams, Dict] = None,
                      _remote: Optional[bool] = None) -> np.ndarray:

        class_probs = _predict(
            self,
            X=X,
            output_margin=self.objective == "multi:softmax",
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
            ray_params=ray_params,
            _remote=_remote,
        )
        # If model is loaded from a raw booster there's no `n_classes_`
        return _cls_predict_proba(
            getattr(self, "n_classes_", None), class_probs, np.vstack)

    def load_model(self, fname):
        if not hasattr(self, "_Booster"):
            self._Booster = Booster()
        return super().load_model(fname)

    predict_proba.__doc__ = (
        XGBClassifier.predict_proba.__doc__ + _RAY_PARAMS_DOC)


RayXGBClassifier.__doc__ = _treat_estimator_doc(XGBClassifier.__doc__)
