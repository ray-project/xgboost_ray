from ray import tune

from xgboost_ray.session import get_actor_rank, put_queue


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
