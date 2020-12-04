# Tune imports.
from typing import Dict

import logging

from xgboost_ray.session import put_queue

try:
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


class TuneReportCallback(OrigTuneReportCallback):
    def __call__(self, env):
        report_dict = self._get_report_dict(env)
        put_queue(lambda: tune.report(**report_dict))


class _TuneCheckpointCallback(_OrigTuneCheckpointCallback):
    def __call__(self, env):
        put_queue(lambda: self._create_checkpoint(env, self._filename))


class TuneReportCheckpointCallback(OrigTuneReportCheckpointCallback):
    _checkpoint_callback_cls = _TuneCheckpointCallback
    _report_callbacks_cls = TuneReportCallback


def _try_add_tune_callback(kwargs: Dict):
    if TUNE_INSTALLED and is_session_enabled():
        callbacks = kwargs.get("callbacks", [])
        new_callbacks = []
        has_tune_callback = False

        REPLACE_MSG = "Replaced `{orig}` with `{target}`. If you want to " \
                      "avoid this warning, pass `{target}` as a callback" \
                      "directly in your calls to `xgboost_ray.train()`."

        for cb in callbacks:
            if isinstance(cb, OrigTuneReportCallback):
                replace_cb = TuneReportCallback(metrics=cb._metrics)
                new_callbacks.append(replace_cb)
                logging.warning(
                    REPLACE_MSG.format(
                        orig="ray.tune.integration.xgboost.TuneReportCallback",
                        targets="xgboost_ray.tune.TuneReportCallback"))
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
                        targets="xgboost_ray.tune.TuneReportCheckpointCallback"
                    ))
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
