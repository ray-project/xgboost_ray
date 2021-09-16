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

import warnings
from typing import Callable, Tuple, Optional, Any, List

import warnings
import functools

from xgboost_ray.main import (RayParams, train, predict, XGBOOST_VERSION_TUPLE,
                              LEGACY_WARNING)
from xgboost_ray.matrix import RayDMatrix

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
