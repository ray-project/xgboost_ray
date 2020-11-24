try:
    from ray import tune
    TUNE_INSTALLED = True
except ImportError:
    TUNE_INSTALLED = False

from xgboost_ray.session import get_actor_rank, put_queue
from xgboost_ray import train


# At each boosting round, add the results to the queue actor.
# The results in the queue will be consumed by the Trainable to report to tune.
# TODO: Subclass ray.tune.integrations.xgboost.TuneReportCallback?
class RayTuneReportCallback:
    def __init__(self, metrics):
        if isinstance(metrics, str):
            metrics = [metrics]
        self._metrics = metrics

    def __call__(self, env):
        if get_actor_rank() == 0:
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
            put_queue(lambda: tune.report(**report_dict))


def hyperparameter_search(*args, metrics=None, **kwargs):
    """Distributed XGBoost hyperparameter tuning via Ray and Ray Tune.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them train an
    XGBoost classifier. The XGBoost parameters will be shared and combined
    via Rabit's all-reduce protocol. This function will also handle
    reporting results to Ray Tune for hyperparameter search.

    Args:
        Same as ``xgboost_ray.train``.
        metrics (str|list|dict): Metrics to report to Tune. If this is a list,
            each item describes the metric key reported to XGBoost,
            and it will reported under the same name to Tune. If this is a
            dict, each key will be the name reported to Tune and the respective
            value will be the metric key reported to XGBoost. If this is None,
            all metrics will be reported to Tune under their default names as
            obtained from XGBoost.

    Returns: An ``xgboost.Booster`` object.
    """
    if not TUNE_INSTALLED:
        raise ImportError("To use Tune with XGBoost, Tune dependencies need "
                          "to be installed. Please run `pip install ray["
                          "tune]` and try again.")
    callbacks = kwargs.get("callbacks", [])
    callbacks.append(RayTuneReportCallback(metrics=metrics))
    return train(*args, **kwargs, callbacks=callbacks)
