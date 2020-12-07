# Tune imports.
import os
from typing import Dict, Union, List

import logging

from xgboost_ray.session import put_queue

try:
    from ray import tune
    from ray.tune import is_session_enabled
    from ray.tune.integration.xgboost import \
        TuneReportCallback as OrigTuneReportCallback, \
        _TuneCheckpointCallback as _OrigTuneCheckpointCallback, \
        TuneReportCheckpointCallback as OrigTuneReportCheckpointCallback
    TUNE_INSTALLED = True
except ImportError:
    tune = TuneReportCallback = _TuneCheckpointCallback = \
        TuneReportCheckpointCallback = None
    OrigTuneReportCallback = _OrigTuneCheckpointCallback = \
        OrigTuneReportCheckpointCallback = object

    def is_session_enabled():
        return False

    TUNE_INSTALLED = False

# Todo(krfricke): Remove after next ray core release
if not hasattr(OrigTuneReportCallback, "_get_report_dict"):
    TUNE_LEGACY = True
else:
    TUNE_LEGACY = False

if TUNE_LEGACY:
    # Until the next release, keep compatible callbacks here.
    class TuneReportCallback(OrigTuneReportCallback):
        def _get_report_dict(self, env):
            # Only one worker should report to Tune
            result_dict = dict(env.evaluation_result_list)
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

        def __call__(self, env):
            report_dict = self._get_report_dict(env)
            put_queue(lambda: tune.report(**report_dict))

    class _TuneCheckpointCallback(_OrigTuneCheckpointCallback):
        def __init__(self, filename: str, frequency: int):
            super(_TuneCheckpointCallback, self).__init__(filename)
            self._frequency = frequency

        @staticmethod
        def _create_checkpoint(env, filename: str, frequency: int):
            if env.iteration % frequency > 0:
                return
            with tune.checkpoint_dir(step=env.iteration) as checkpoint_dir:
                env.model.save_model(os.path.join(checkpoint_dir, filename))

        def __call__(self, env):
            put_queue(lambda: self._create_checkpoint(
                env, filename=self._filename, frequency=self._frequency))

    class TuneReportCheckpointCallback(OrigTuneReportCheckpointCallback):
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

        def __call__(self, env):
            self._checkpoint(env)
            self._report(env)

else:
    # New style callbacks.
    class TuneReportCallback(OrigTuneReportCallback):
        def __call__(self, env):
            report_dict = self._get_report_dict(env)
            put_queue(lambda: tune.report(**report_dict))

    class _TuneCheckpointCallback(_OrigTuneCheckpointCallback):
        def __call__(self, env):
            put_queue(lambda: self._create_checkpoint(
                env, filename=self._filename, frequency=self._frequency))

    class TuneReportCheckpointCallback(OrigTuneReportCheckpointCallback):
        _checkpoint_callback_cls = _TuneCheckpointCallback
        _report_callbacks_cls = TuneReportCallback


def _try_add_tune_callback(kwargs: Dict):
    if TUNE_INSTALLED and is_session_enabled():
        callbacks = kwargs.get("callbacks", [])
        new_callbacks = []
        has_tune_callback = False

        REPLACE_MSG = "Replaced `{orig}` with `{target}`. If you want to " \
                      "avoid this warning, pass `{target}` as a callback " \
                      "directly in your calls to `xgboost_ray.train()`."

        for cb in callbacks:
            if isinstance(cb, OrigTuneReportCallback):
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
            elif isinstance(
                    cb, (TuneReportCallback, TuneReportCheckpointCallback)):
                has_tune_callback = True
                new_callbacks.append(cb)
            else:
                new_callbacks.append(cb)

        if not has_tune_callback:
            # Todo: Maybe add checkpointing callback
            new_callbacks.append(TuneReportCallback())

        kwargs["callbacks"] = new_callbacks
