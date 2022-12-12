# Tune imports.
from typing import Dict, Optional

import ray

import logging

from ray.util.annotations import PublicAPI

from xgboost_ray.xgb import xgboost as xgb

from xgboost_ray.session import put_queue, get_rabit_rank
from xgboost_ray.util import Unavailable, force_on_current_node

try:
    from ray import tune
    from ray.tune import is_session_enabled
    from ray.tune.utils import flatten_dict
    from ray.tune.integration.xgboost import \
        TuneReportCallback as OrigTuneReportCallback, \
        _TuneCheckpointCallback as _OrigTuneCheckpointCallback, \
        TuneReportCheckpointCallback as OrigTuneReportCheckpointCallback

    TUNE_INSTALLED = True
except ImportError:
    tune = None
    TuneReportCallback = _TuneCheckpointCallback = \
        TuneReportCheckpointCallback = Unavailable
    OrigTuneReportCallback = _OrigTuneCheckpointCallback = \
        OrigTuneReportCheckpointCallback = object

    def is_session_enabled():
        return False

    flatten_dict = is_session_enabled
    TUNE_INSTALLED = False

if TUNE_INSTALLED:
    # New style callbacks.
    class TuneReportCallback(OrigTuneReportCallback):
        def after_iteration(self, model, epoch: int, evals_log: Dict):
            if get_rabit_rank() == 0:
                report_dict = self._get_report_dict(evals_log)
                put_queue(lambda: tune.report(**report_dict))

    class _TuneCheckpointCallback(_OrigTuneCheckpointCallback):
        def after_iteration(self, model, epoch: int, evals_log: Dict):
            if get_rabit_rank() == 0:
                put_queue(lambda: self._create_checkpoint(
                    model, epoch, self._filename, self._frequency))

    class TuneReportCheckpointCallback(OrigTuneReportCheckpointCallback):
        _checkpoint_callback_cls = _TuneCheckpointCallback
        _report_callbacks_cls = TuneReportCallback


def _try_add_tune_callback(kwargs: Dict):
    if TUNE_INSTALLED and is_session_enabled():
        callbacks = kwargs.get("callbacks", []) or []
        new_callbacks = []
        has_tune_callback = False

        REPLACE_MSG = "Replaced `{orig}` with `{target}`. If you want to " \
                      "avoid this warning, pass `{target}` as a callback " \
                      "directly in your calls to `xgboost_ray.train()`."

        for cb in callbacks:
            if isinstance(cb,
                          (TuneReportCallback, TuneReportCheckpointCallback)):
                has_tune_callback = True
                new_callbacks.append(cb)
            elif isinstance(cb, OrigTuneReportCallback):
                replace_cb = TuneReportCallback(metrics=cb._metrics)
                new_callbacks.append(replace_cb)
                logging.warning(
                    REPLACE_MSG.format(
                        orig="ray.tune.integration.xgboost.TuneReportCallback",
                        target="xgboost_ray.tune.TuneReportCallback"))
                has_tune_callback = True
            elif isinstance(cb, OrigTuneReportCheckpointCallback):
                replace_cb = TuneReportCheckpointCallback(
                    metrics=cb._report._metrics,
                    filename=cb._checkpoint._filename,
                    frequency=cb._checkpoint._frequency)
                new_callbacks.append(replace_cb)
                logging.warning(
                    REPLACE_MSG.format(
                        orig="ray.tune.integration.xgboost."
                        "TuneReportCheckpointCallback",
                        target="xgboost_ray.tune.TuneReportCheckpointCallback")
                )
                has_tune_callback = True
            else:
                new_callbacks.append(cb)

        if not has_tune_callback:
            # Todo: Maybe add checkpointing callback
            new_callbacks.append(TuneReportCallback())

        kwargs["callbacks"] = new_callbacks
        return True
    else:
        return False


def _get_tune_resources(num_actors: int, cpus_per_actor: int,
                        gpus_per_actor: int,
                        resources_per_actor: Optional[Dict],
                        placement_options: Optional[Dict]):
    """Returns object to use for ``resources_per_trial`` with Ray Tune."""
    if TUNE_INSTALLED:
        from ray.tune import PlacementGroupFactory

        head_bundle = {}
        child_bundle = {"CPU": cpus_per_actor, "GPU": gpus_per_actor}
        child_bundle_extra = {} if resources_per_actor is None else \
            resources_per_actor
        child_bundles = [{
            **child_bundle,
            **child_bundle_extra
        } for _ in range(num_actors)]
        bundles = [head_bundle] + child_bundles
        placement_options = placement_options or {}
        placement_options.setdefault("strategy", "PACK")
        # Special case, same as in
        # ray.air.ScalingConfig.as_placement_group_factory
        # TODO remove after Ray 2.3 is out
        if placement_options.get("_max_cpu_fraction_per_node", None) is None:
            placement_options.pop("_max_cpu_fraction_per_node", None)
        placement_group_factory = PlacementGroupFactory(
            bundles, **placement_options)

        return placement_group_factory
    else:
        raise RuntimeError("Tune is not installed, so `get_tune_resources` is "
                           "not supported. You can install Ray Tune via `pip "
                           "install ray[tune]`.")


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
