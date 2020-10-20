from threading import Thread
from typing import Tuple, Dict, Any, List

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

import xgboost as xgb


from xgboost_ray.matrix import RayDMatrix


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


@ray.remote
class RayXGBoostActor:
    def __init__(self, rank: int, num_actors: int):
        self.rank = rank
        self.num_actors = num_actors

        self._data: Dict[RayDMatrix, xgb.DMatrix] = {}
        self._evals = []

    def load_data(self, data: RayDMatrix):
        x, y = ray.get(data.load_data(self.rank, self.num_actors))
        matrix = xgb.DMatrix(x, label=y)
        self._data[data] = matrix

    def train(self,
              rabit_args: List[str],
              params: Dict[str, Any],
              dtrain: RayDMatrix,
              evals: Tuple[RayDMatrix, str],
              *args,
              **kwargs):
        local_params = params.copy()

        if dtrain not in self._data:
            self.load_data(dtrain)
        local_dtrain = self._data[dtrain]

        local_evals = []
        for deval, name in evals:
            if deval not in self._data:
                self.load_data(deval)
            local_evals.append((self._data[deval], name))

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


def train(
        params: Dict,
        dtrain: RayDMatrix,
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
        RayXGBoostActor.options(num_gpus=gpus_per_worker).remote(
            rank=i, num_actors=num_actors)
        for i in range(num_actors)
    ]
    logger.info(f"[RayXGBoost] Created {len(actors)} remote actors.")

    # Split data across workers
    wait_load = []
    for i, actor in enumerate(actors):
        wait_load.append(actor.load_data.remote(dtrain))
        for deval, name in evals:
            wait_load.append(actor.load_data.remote(deval))

    ray.get(wait_load)

    logger.info("[RayXGBoost] Starting XGBoost training.")

    # Start tracker
    env = _start_rabit_tracker(num_actors)
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]

    # Train
    fut = [
        actor.train.remote(rabit_args, params, dtrain, evals, *args, **kwargs)
        for actor in actors
    ]

    # All results should be the same because of Rabit tracking. So we just
    # return the first one.
    res: Dict[str, Any] = ray.get(fut[0])
    bst = res["bst"]
    evals_result = res["evals_result"]

    return bst, evals_result
