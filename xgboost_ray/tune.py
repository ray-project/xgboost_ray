from ray import tune

from xgboost_ray.session import get_actor_rank, put_queue


class RayTuneReportCallback:
    """xgboost-ray to Ray Tune reporting callback

    Reports metrics to Ray Tune. When calling xgboost_ray.train inside a
    Tune session, this callback is passed to xgboost. At the end of each
    boosting round, the rank 0 worker sends its results to the
    worker-driver communication Queue, and the results are reported to
    tune from the driver via tune.report. The callback is a no-op for all
    other workers. Only the rank 0 worker needs to send results to the
    queue since Rabit all-reduce ensures that all workers are in sync.


    """

    def __call__(self, env):
        if get_actor_rank() == 0:
            result_dict = dict(env.evaluation_result_list)
            put_queue(lambda: tune.report(**result_dict))
