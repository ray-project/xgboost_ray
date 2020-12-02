# Tune imports.
from typing import Dict

try:
    from ray.tune import is_session_enabled
    from ray.tune.integration.xgboost import TuneReportCallback, \
        TuneReportCheckpointCallback
    TUNE_INSTALLED = True
except ImportError:
    tune = TuneReportCallback = TuneReportCheckpointCallback = None

    def is_session_enabled():
        return False

    TUNE_INSTALLED = False


def _try_add_tune_callback(kwargs: Dict):
    if TUNE_INSTALLED and is_session_enabled():
        callbacks = kwargs.get("callbacks", [])

        if any(
                isinstance(cb, (TuneReportCallback,
                                TuneReportCheckpointCallback))
                for cb in callbacks):
            return

        callbacks.append(TuneReportCallback())
        kwargs["callbacks"] = callbacks
