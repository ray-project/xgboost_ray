from typing import Dict, Optional

from ray import tune
from xgboost_ray.session import get_actor_rank, put_queue
from xgboost_ray import train, RayDMatrix

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

def hyperparameter_search(params: Dict,
          dtrain: RayDMatrix,
          *args,
          evals=(),
          evals_result: Optional[Dict] = None,
          additional_results: Optional[Dict] = None,
          num_actors: int = 4,
          cpus_per_actor: int = 0,
          gpus_per_actor: int = -1,
          resources_per_actor: Optional[Dict] = None,
          max_actor_restarts: int = 0,
          **kwargs):
    """Distributed XGBoost tuning via Ray and Ray Tune.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them train an
    XGBoost classifier. The XGBoost parameters will be shared and combined
    via Rabit's all-reduce protocol. This function will also handle
    reporting results to Ray Tune for hyperparameter search.

    Args:
        params (Dict): parameter dict passed to ``xgboost.train()``
        dtrain (RayDMatrix): Data object containing the training data.
        evals (Union[List[Tuple], Tuple]): ``evals`` tuple passed to
            ``xgboost.train()``.
        evals_result (Optional[Dict]): Dict to store evaluation results in.
        additional_results (Optional[Dict]): Dict to store additional results.
        num_actors (int): Number of parallel Ray actors.
        cpus_per_actor (int): Number of CPUs to be used per Ray actor.
        gpus_per_actor (int): Number of GPUs to be used per Ray actor.
        resources_per_actor (Optional[Dict]): Dict of additional resources
            required per Ray actor.
        max_actor_restarts (int): Number of retries when Ray actors fail.
            Defaults to 0 (no retries). Set to -1 for unlimited retries.

    Keyword Args:
        checkpoint_prefix (str): Prefix for the checkpoint filenames.
            Defaults to ``.xgb_ray_{time.time()}``.
        checkpoint_path (str): Path to store checkpoints at. Defaults to
            ``/tmp``
        checkpoint_frequency (int): How often to save checkpoints. Defaults
            to ``5``.

    Returns: An ``xgboost.Booster`` object.
    """
    callbacks =kwargs.get("callbacks", [])
    callbacks.append(RayTuneReportCallback(metrics=None))
    train(params, dtrain, *args, evals=evals, evals_result=evals_result,
          additional_results=additional_results,
          num_actors=num_actors, cpus_per_actor=cpus_per_actor,
          gpus_per_actor=gpus_per_actor,
          resources_per_actor=resources_per_actor,
          max_actor_restarts=max_actor_restarts, callbacks=callbacks, **kwargs)