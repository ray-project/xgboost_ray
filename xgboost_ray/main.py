from threading import Thread
from typing import Tuple, Dict, Any, Union, Optional

try:
    import ray
    from ray import logger
    from ray.services import get_node_ip_address
    RAY_INSTALLED = True
except ImportError:
    ray = None
    logger = None
    get_node_ip_address = None
    RAY_INSTALLED = False

import pandas as pd
import numpy as np
import xgboost as xgb


Data = Union[ np.ndarray, pd.DataFrame, pd.Series]


def _assert_ray_support():
    if not RAY_INSTALLED:
        raise ImportError(
            'Ray needs to be installed in order to use this module. '
            'Try: `pip install ray`')


def _start_rabit_tracker(num_workers: int):
    """Start Rabit tracker. The workers connect to this tracker to share
    their results."""
    host = get_node_ip_address()

    env = {'DMLC_NUM_WORKER': num_workers}
    rabit_tracker = xgb.RabitTracker(hostIP=host, nslave=num_workers)

    # Get tracker Host + IP
    env.update(rabit_tracker.slave_envs())
    rabit_tracker.start(num_workers)

    # Wait until context completion
    thread = Thread(target=rabit_tracker.join)
    thread.daemon = True
    thread.start()

    return env


class RabitContext:
    """Context to connect a worker to a rabit tracker"""
    def __init__(self, actor_id, args):
        self.args = args
        self.args.append(
            ('DMLC_TASK_ID=[xgboost.ray]:' + actor_id).encode())

    def __enter__(self):
        xgb.rabit.init(self.args)

    def __exit__(self, *args):
        xgb.rabit.finalize()


class RayDMatrix:
    def __init__(self, X: Data, y: Optional[Data] = None):
        self.X = X
        self.y = y

    def __iter__(self):
        yield self.X
        yield self.y


@ray.remote
class RayXGBoostActor:
    def __init__(self):
        self._dtrain = []
        self._evals = []

    def set_X_y(self, X, y):
        self._dtrain = xgb.DMatrix(X, y)

    def add_eval_X_y(self, X, y, method):
        self._evals.append((xgb.DMatrix(X, y), method))

    def train(self, rabit_args, params, *args, **kwargs):
        local_params = params.copy()
        local_dtrain = self._dtrain
        local_evals = self._evals

        evals_result = dict()

        with RabitContext(str(id(self)), rabit_args):
            bst = xgb.train(
                local_params,
                local_dtrain,
                *args,
                evals=local_evals,
                evals_result=evals_result,
                **kwargs
            )
            return {"bst": bst, "evals_result": evals_result}


def _data_slice(data, indices):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.loc[indices]
    elif isinstance(data, np.ndarray):
        return data[indices]
    else:
        raise NotImplementedError


def train(
        params: Dict,
        data: Union[RayDMatrix, Tuple[Data, Data]],
        *args, evals=(),
        num_actors: int = 4,
        gpus_per_worker: int = -1,
        **kwargs):
    _assert_ray_support()

    if not ray.is_initialized():
        ray.init()

    if gpus_per_worker == -1:
        gpus_per_worker = 0
        if "tree_method" in params and params["tree_method"].startswith("gpu"):
            gpus_per_worker = 1

    # Create remote actors
    actors = [
        RayXGBoostActor.options(num_gpus=gpus_per_worker).remote()
        for _ in range(num_actors)
    ]
    logger.info(f"[RayXGBoost] Created {len(actors)} remote actors.")

    X, y = data
    assert len(X) == len(y)

    # Split data across workers
    for i, actor in enumerate(actors):
        indices = range(i, len(X), len(actors))

        X_ref = ray.put(_data_slice(X, indices))
        y_ref = ray.put(_data_slice(y, indices))
        actor.set_X_y.remote(X_ref, y_ref)

        for i, ((eval_X, eval_y), eval_method) in enumerate(evals):
            eval_indices = range(i, len(eval_X), len(actors))
            eval_X_ref = ray.put(_data_slice(eval_X, eval_indices))
            eval_y_ref = ray.put(_data_slice(eval_y, eval_indices))

            actor.add_eval_X_y.remote(eval_X_ref, eval_y_ref, eval_method)

    logger.info("[RayXGBoost] Starting XGBoost training.")

    # Start tracker
    env = _start_rabit_tracker(num_actors)
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]

    # Train
    fut = [actor.train.remote(rabit_args, params, *args, **kwargs) for actor in actors]

    # All results should be the same because of Rabit tracking. So we just
    # return the first one.
    res: Dict[str, Any] = ray.get(fut[0])
    bst = res["bst"]
    evals_result = res["evals_result"]

    return bst, evals_result
