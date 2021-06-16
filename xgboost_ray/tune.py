# Tune imports.
import os
from typing import Dict, Union, List, Optional

import ray

try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict

import logging

import xgboost as xgb

from xgboost_ray.compat import TrainingCallback
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

# Todo(krfricke): Remove after next ray core release
if not hasattr(OrigTuneReportCallback, "_get_report_dict") or not issubclass(
        OrigTuneReportCallback, TrainingCallback):
    TUNE_LEGACY = True
else:
    TUNE_LEGACY = False

# Todo(amogkam): Remove after Ray 1.3 release.
try:
    from ray.tune import PlacementGroupFactory

    TUNE_USING_PG = True
except ImportError:
    TUNE_USING_PG = False
    PlacementGroupFactory = Unavailable

if TUNE_LEGACY and TUNE_INSTALLED:
    # Until the next release, keep compatible callbacks here.
    class TuneReportCallback(OrigTuneReportCallback, TrainingCallback):
        def _get_report_dict(self, evals_log):
            if isinstance(evals_log, OrderedDict):
                # xgboost>=1.3
                result_dict = flatten_dict(evals_log, delimiter="-")
                for k in list(result_dict):
                    result_dict[k] = result_dict[k][0]
            else:
                # xgboost<1.3
                result_dict = dict(evals_log)
            if not self._metrics:
                report_dict = result_dict
            else:
                report_dict = {}
                for key in self._metrics:
                    if isinstance(self._metrics, dict):
                        metric = self._metrics[key]
                    else:
                        metric = key
                    report_dict[key] = result_dict[metric]
            return report_dict

        def after_iteration(self, model, epoch: int, evals_log: Dict):
            if get_rabit_rank() == 0:
                report_dict = self._get_report_dict(evals_log)
                put_queue(lambda: tune.report(**report_dict))

    class _TuneCheckpointCallback(_OrigTuneCheckpointCallback,
                                  TrainingCallback):
        def __init__(self, filename: str, frequency: int):
            super(_TuneCheckpointCallback, self).__init__(filename)
            self._frequency = frequency

        @staticmethod
        def _create_checkpoint(model, epoch: int, filename: str,
                               frequency: int):
            if epoch % frequency > 0:
                return
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                model.save_model(os.path.join(checkpoint_dir, filename))

        def after_iteration(self, model, epoch: int, evals_log: Dict):
            if get_rabit_rank() == 0:
                put_queue(lambda: self._create_checkpoint(
                    model, epoch, self._filename, self._frequency))

    class TuneReportCheckpointCallback(OrigTuneReportCheckpointCallback,
                                       TrainingCallback):
        _checkpoint_callback_cls = _TuneCheckpointCallback
        _report_callbacks_cls = TuneReportCallback

        def __init__(
                self,
                metrics: Union[None, str, List[str], Dict[str, str]] = None,
                filename: str = "checkpoint",
                frequency: int = 5):
            self._checkpoint = self._checkpoint_callback_cls(
                filename, frequency)
            self._report = self._report_callbacks_cls(metrics)

        def after_iteration(self, model, epoch: int, evals_log: Dict):
            if get_rabit_rank() == 0:
                self._checkpoint.after_iteration(model, epoch, evals_log)
                self._report.after_iteration(model, epoch, evals_log)

elif TUNE_INSTALLED:
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
                if TUNE_LEGACY:
                    replace_cb = TuneReportCheckpointCallback(
                        metrics=cb._report._metrics,
                        filename=cb._checkpoint._filename)
                else:
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
                        resources_per_actor: Optional[Dict]):
    """Returns object to use for ``resources_per_trial`` with Ray Tune."""
    if TUNE_INSTALLED:
        if not TUNE_USING_PG:
            resources_per_actor = {} if not resources_per_actor \
                else resources_per_actor
            extra_custom_resources = {
                k: v * num_actors
                for k, v in resources_per_actor.items()
            }
            return dict(
                cpu=1,
                extra_cpu=cpus_per_actor * num_actors,
                extra_gpu=gpus_per_actor * num_actors,
                extra_custom_resources=extra_custom_resources,
            )
        else:
            from ray.tune import PlacementGroupFactory

            head_bundle = {"CPU": 1}
            child_bundle = {"CPU": cpus_per_actor, "GPU": gpus_per_actor}
            child_bundle_extra = {} if resources_per_actor is None else \
                resources_per_actor
            child_bundles = [{
                **child_bundle,
                **child_bundle_extra
            } for _ in range(num_actors)]
            bundles = [head_bundle] + child_bundles
            placement_group_factory = PlacementGroupFactory(
                bundles, strategy="PACK")

            return placement_group_factory
    else:
        raise RuntimeError("Tune is not installed, so `get_tune_resources` is "
                           "not supported. You can install Ray Tune via `pip "
                           "install ray[tune]`.")


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
