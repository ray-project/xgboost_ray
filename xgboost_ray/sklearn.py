from typing import Tuple, Dict, Any, List, Optional, Callable, Union, Sequence

import numpy as np

import warnings
import copy
import json

from xgboost import Booster
from xgboost.sklearn import XGBModel, XGBClassifier, XGBRegressor, _deprecate_positional_args, _objective_decorator, _wrap_evaluation_matrices, _convert_ntree_limit, _is_cudf_df, _is_cudf_ser, _is_cupy_array, _cls_predict_proba
from xgboost.compat import XGBoostLabelEncoder

from xgboost_ray.main import RayParams, train, predict
from xgboost_ray.matrix import RayDMatrix


# would normally use a mixin class but it breaks xgb's get_params
def _predict(
        model: XGBModel,
        X,
        output_margin=False,
        ntree_limit=None,
        validate_features=True,
        base_margin=None,
        iteration_range=None,
):
    iteration_range = _convert_ntree_limit(model.get_booster(), ntree_limit,
                                           iteration_range)
    iteration_range = model._get_iteration_range(iteration_range)

    test = RayDMatrix(X, base_margin=base_margin, missing=model.missing)
    return predict(
        model.get_booster(),
        data=test,
        iteration_range=iteration_range,
        output_margin=output_margin,
        validate_features=validate_features,
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
            callbacks=None):
        evals_result = {}

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

        ray_params = RayParams(num_actors=self.n_jobs or 1)

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
        )

        self._set_evaluation_result(evals_result)
        return self

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
    ):
        return _predict(
            self,
            X,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range)

    def load_model(self, fname):
        if not hasattr(self, '_Booster'):
            self._Booster = Booster({'n_jobs': 1})
        return super().load_model(fname)


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
            callbacks=None):
        # pylint: disable = attribute-defined-outside-init,too-many-statements
        can_use_label_encoder = True
        label_encoding_check_error = (
            "The label must consist of integer "
            "labels of form 0, 1, 2, ..., [num_class - 1].")
        label_encoder_deprecation_msg = (
            "The use of label encoder in XGBClassifier is deprecated and will be "
            "removed in a future release. To remove this warning, do the "
            "following: 1) Pass option use_label_encoder=False when constructing "
            "XGBClassifier object; and 2) Encode your labels (y) as integers "
            "starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].")

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
                    'The option use_label_encoder=True is incompatible with inputs '
                    +
                    'of type cuDF or cuPy. Please set use_label_encoder=False when '
                    + 'constructing XGBClassifier object. NOTE: ' +
                    label_encoder_deprecation_msg)
            warnings.warn(label_encoder_deprecation_msg, UserWarning)
            self._le = XGBoostLabelEncoder().fit(y)
            label_transform = self._le.transform
        else:
            label_transform = lambda x: x

        model, feval, params = self._configure_fit(xgb_model, eval_metric,
                                                   params)
        if len(X.shape) != 2:
            # Simply raise an error here since there might be many
            # different ways of reshaping
            raise ValueError(
                "Please reshape the input data X into 2-dimensional matrix.")

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

        ray_params = RayParams(num_actors=self.n_jobs or 1)

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
        )

        if not callable(self.objective):
            self.objective = params["objective"]

        self._set_evaluation_result(evals_result)
        return self

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
    ):
        class_probs = _predict(
            self,
            X=X,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
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

        if hasattr(self, '_le'):
            return self._le.inverse_transform(column_indexes)
        return column_indexes

    def predict_proba(
            self,
            X,
            ntree_limit=None,
            validate_features=False,
            base_margin=None,
            iteration_range: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """ Predict the probability of each `X` example being of a given class.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X : array_like
            Feature matrix.
        ntree_limit : int
            Deprecated, use `iteration_range` instead.
        validate_features : bool
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin : array_like
            Margin added to prediction.
        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

        Returns
        -------
        prediction : numpy array
            a numpy array of shape array-like of shape (n_samples, n_classes) with the
            probability of each data example being of a given class.
        """
        # custom obj:      Do nothing as we don't know what to do.
        # softprob:        Do nothing, output is proba.
        # softmax:         Use output margin to remove the argmax in PredTransform.
        # binary:logistic: Expand the prob vector into 2-class matrix after predict.
        # binary:logitraw: Unsupported by predict_proba()
        class_probs = _predict(
            self,
            X=X,
            output_margin=self.objective == "multi:softmax",
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range)
        # If model is loaded from a raw booster there's no `n_classes_`
        return _cls_predict_proba(
            getattr(self, "n_classes_", None), class_probs, np.vstack)

    def load_model(self, fname):
        if not hasattr(self, '_Booster'):
            self._Booster = Booster({'n_jobs': 1})
        return super().load_model(fname)
