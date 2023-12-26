import logging
from typing import Dict, Optional

import ray
from ray.util.annotations import PublicAPI

from xgboost_ray.session import get_rabit_rank, put_queue
from xgboost_ray.util import force_on_current_node
from xgboost_ray.xgb import xgboost as xgb

try:
    from ray import train, tune  # noqa: F401
except (ImportError, ModuleNotFoundError) as e:
    raise RuntimeError(
        "Ray Train and Ray Tune are required dependencies of xgboost_ray. "
        'Please install with: `pip install "ray[train]"`'
    ) from e

import ray.train
from ray.tune.integration.xgboost import TuneReportCallback as OrigTuneReportCallback
from ray.tune.integration.xgboost import (
    TuneReportCheckpointCallback as OrigTuneReportCheckpointCallback,
)


class TuneReportCheckpointCallback(OrigTuneReportCheckpointCallback):
    def after_iteration(self, model, epoch: int, evals_log: Dict):
        # NOTE: We need to update `evals_log` here (even though the super method
        # already does it) because the actual callback method gets run
        # in a different process, so *this* instance of the callback will not have
        # access to the `evals_log` dict in `after_training`.
        self._evals_log = evals_log

        if get_rabit_rank() == 0:
            put_queue(
                lambda: super(TuneReportCheckpointCallback, self).after_iteration(
                    model=model, epoch=epoch, evals_log=evals_log
                )
            )

    def after_training(self, model):
        if get_rabit_rank() == 0:
            put_queue(
                lambda: super(TuneReportCheckpointCallback, self).after_training(
                    model=model
                )
            )
        return model


class TuneReportCallback(OrigTuneReportCallback):
    def __new__(cls: type, *args, **kwargs):
        # TODO(justinvyu): [code_removal] Remove in Ray 2.11.
        raise DeprecationWarning(
            "`TuneReportCallback` is deprecated. "
            "Use `xgboost_ray.tune.TuneReportCheckpointCallback` instead."
        )


def _try_add_tune_callback(kwargs: Dict):
    ray_train_context_initialized = bool(ray.train.get_context())
    if ray_train_context_initialized:
        callbacks = kwargs.get("callbacks", []) or []
        new_callbacks = []
        has_tune_callback = False

        REPLACE_MSG = (
            "Replaced `{orig}` with `{target}`. If you want to "
            "avoid this warning, pass `{target}` as a callback "
            "directly in your calls to `xgboost_ray.train()`."
        )

        for cb in callbacks:
            if isinstance(cb, TuneReportCheckpointCallback):
                has_tune_callback = True
                new_callbacks.append(cb)
            elif isinstance(cb, OrigTuneReportCheckpointCallback):
                orig_metrics = cb._metrics
                orig_frequency = cb._frequency

                replace_cb = TuneReportCheckpointCallback(
                    metrics=orig_metrics, frequency=orig_frequency
                )
                new_callbacks.append(replace_cb)
                logging.warning(
                    REPLACE_MSG.format(
                        orig="ray.tune.integration.xgboost."
                        "TuneReportCheckpointCallback",
                        target="xgboost_ray.tune.TuneReportCheckpointCallback",
                    )
                )
                has_tune_callback = True
            else:
                new_callbacks.append(cb)

        if not has_tune_callback:
            new_callbacks.append(TuneReportCheckpointCallback(frequency=0))

        kwargs["callbacks"] = new_callbacks
        return True
    else:
        return False


def _get_tune_resources(
    num_actors: int,
    cpus_per_actor: int,
    gpus_per_actor: int,
    resources_per_actor: Optional[Dict],
    placement_options: Optional[Dict],
):
    """Returns object to use for ``resources_per_trial`` with Ray Tune."""
    from ray.tune import PlacementGroupFactory

    head_bundle = {}
    child_bundle = {"CPU": cpus_per_actor, "GPU": gpus_per_actor}
    child_bundle_extra = {} if resources_per_actor is None else resources_per_actor
    child_bundles = [{**child_bundle, **child_bundle_extra} for _ in range(num_actors)]
    bundles = [head_bundle] + child_bundles
    placement_options = placement_options or {}
    placement_options.setdefault("strategy", "PACK")
    placement_group_factory = PlacementGroupFactory(bundles, **placement_options)

    return placement_group_factory


@PublicAPI(stability="beta")
def load_model(model_path):
    """Loads the model stored in the provided model_path.

    If using Ray Client, this will automatically handle loading the path on
    the server by using a Ray task.

    Returns:
        xgb.Booster object of the model stored in the provided model_path

    """

    def load_model_fn(model_path):
        best_bst = xgb.Booster()
        best_bst.load_model(model_path)
        return best_bst

    # Load the model checkpoint.
    if ray.util.client.ray.is_connected():
        # If using Ray Client, the best model is saved on the server.
        # So we have to wrap the model loading in a ray task.
        remote_load = ray.remote(load_model_fn)
        remote_load = force_on_current_node(remote_load)
        bst = ray.get(remote_load.remote(model_path))
    else:
        bst = load_model_fn(model_path)

    return bst
