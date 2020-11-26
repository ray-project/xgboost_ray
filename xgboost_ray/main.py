from typing import Tuple, Dict, Any, List, Optional, Callable

import numpy as np

import os
import time

from threading import Thread

try:
    import ray
    from ray import logger
    from ray.services import get_node_ip_address
    from ray.exceptions import RayActorError
    from ray.util.queue import Queue
    from ray.actor import ActorHandle
    RAY_INSTALLED = True
except ImportError:
    ray = None
    logger = None
    get_node_ip_address = None
    Queue = None
    ActorHandle = None
    RAY_INSTALLED = False

# Tune imports.
try:
    from ray import tune
    from xgboost_ray.tune import RayTuneReportCallback
    TUNE_INSTALLED = True
except ImportError:
    tune = None
    RayTuneReportCallback = None
    TUNE_INSTALLED = False

import xgboost as xgb

from xgboost_ray.matrix import RayDMatrix, combine_data, \
    RayDeviceQuantileDMatrix, RayDataIter, concat_dataframes
from xgboost_ray.session import init_session


def _assert_ray_support():
    if not RAY_INSTALLED:
        raise ImportError(
            "Ray needs to be installed in order to use this module. "
            "Try: `pip install ray`")


def _start_rabit_tracker(num_workers: int):
    """Start Rabit tracker. The workers connect to this tracker to share
    their results."""
    host = get_node_ip_address()

    env = {"DMLC_NUM_WORKER": num_workers}
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
        self.args.append(("DMLC_TASK_ID=[xgboost.ray]:" + actor_id).encode())

    def __enter__(self):
        xgb.rabit.init(self.args)

    def __exit__(self, *args):
        xgb.rabit.finalize()


def _ray_get_actor_cpus():
    # Get through resource IDs
    resource_ids = ray.get_resource_ids()
    if "CPU" in resource_ids:
        return sum(cpu[1] for cpu in resource_ids["CPU"])
    return None


def _ray_get_cluster_cpus():
    return ray.cluster_resources().get("CPU", None)


def _get_max_node_cpus():
    max_node_cpus = max(
        node.get("Resources", {}).get("CPU", 0.0) for node in ray.nodes())
    return max_node_cpus if max_node_cpus > 0.0 else _ray_get_cluster_cpus()


def _set_omp_num_threads():
    ray_cpus = _ray_get_actor_cpus()
    if ray_cpus:
        os.environ["OMP_NUM_THREADS"] = str(int(ray_cpus))
    else:
        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]
    return int(float(os.environ.get("OMP_NUM_THREADS", "0.0")))


def _checkpoint_file(path: str, prefix: str, rank: int):
    if not prefix:
        return None
    return os.path.join(path, f"{prefix}_{rank:05d}.xgb")


def _try_add_tune_callback(kwargs: Dict):
    if TUNE_INSTALLED and tune.is_session_enabled():
        callbacks = kwargs.get("callbacks", [])
        for callback in callbacks:
            if isinstance(callback, RayTuneReportCallback):
                return
        callbacks.append(RayTuneReportCallback())
        kwargs["callbacks"] = callbacks


def _get_dmatrix(data: RayDMatrix, param: Dict) -> xgb.DMatrix:
    if isinstance(data, RayDeviceQuantileDMatrix):
        if isinstance(param["data"], list):
            dm_param = {
                "feature_names": data.feature_names,
                "feature_types": data.feature_types,
                "missing": data.missing,
            }
            if not isinstance(data, xgb.DeviceQuantileDMatrix):
                pass
            param.update(dm_param)
            it = RayDataIter(**param)
            matrix = xgb.DeviceQuantileDMatrix(it, **dm_param)
        else:
            matrix = xgb.DeviceQuantileDMatrix(**param)
    else:
        if isinstance(param["data"], list):
            dm_param = {
                "data": concat_dataframes(param["data"]),
                "label": concat_dataframes(param["label"]),
                "weight": concat_dataframes(param["weight"]),
                "base_margin": concat_dataframes(param["base_margin"]),
                "label_lower_bound": concat_dataframes(
                    param["label_lower_bound"]),
                "label_upper_bound": concat_dataframes(
                    param["label_upper_bound"]),
            }
            param.update(dm_param)

        ll = param.pop("label_lower_bound", None)
        lu = param.pop("label_upper_bound", None)

        matrix = xgb.DMatrix(**param)
        matrix.set_info(label_lower_bound=ll, label_upper_bound=lu)
    return matrix


@ray.remote
class RayXGBoostActor:
    """Remote Ray XGBoost actor class.

    This remote actor handles local training and prediction of one data
    shard. It initializes a Rabit context, thus connecting to the Rabit
    all-reduce ring, and initializes local training, sending updates
    to other workers.

    The actor also takes care of checkpointing (locally), enabling
    training to resume after it has been stopped.

    Args:
        rank (int): Rank of the actor. Must be ``0 <= rank < num_actors``.
        num_actors (int): Total number of actors.
        queue (Queue): Ray queue to communicate with main process.
        checkpoint_prefix (str): Prefix for checkpoint files.
        checkpoint_path (str): Path to store checkpoints at. Defaults to
            ``/tmp``
        checkpoint_frequency (int): How often to store checkpoints. Defaults
            to ``5``, saving checkpoints every 5 boosting rounds.

    """

    def __init__(self,
                 rank: int,
                 num_actors: int,
                 queue: Optional[Queue] = None,
                 checkpoint_prefix: Optional[str] = None,
                 checkpoint_path: str = "/tmp",
                 checkpoint_frequency: int = 5):
        self.queue = queue
        init_session(rank, queue)

        self.rank = rank
        self.num_actors = num_actors

        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency

        self._data: Dict[RayDMatrix, xgb.DMatrix] = {}

        self._local_n = 0

        _set_omp_num_threads()

    @property
    def checkpoint_file(self) -> Optional[str]:
        return _checkpoint_file(self.checkpoint_path, self.checkpoint_prefix,
                                self.rank)

    @property
    def _save_checkpoint_callback(self):
        def callback(env):
            if env.iteration % self.checkpoint_frequency == 0:
                env.model.save_model(self.checkpoint_file)

        return callback

    def load_data(self, data: RayDMatrix):
        if data in self._data:
            return
        param = data.get_data(self.rank, self.num_actors)
        if isinstance(param["data"], list):
            self._local_n = sum(len(a) for a in param["data"])
        else:
            self._local_n = len(param["data"])
        data.unload_data()  # Free object store

        matrix = _get_dmatrix(data, param)

        self._data[data] = matrix

    def train(self, rabit_args: List[str], params: Dict[str, Any],
              dtrain: RayDMatrix, evals: Tuple[RayDMatrix, str], *args,
              **kwargs) -> Dict[str, Any]:
        num_threads = _set_omp_num_threads()

        local_params = params.copy()

        if "nthread" not in local_params:
            if num_threads > 0:
                local_params["num_threads"] = num_threads
            else:
                local_params["nthread"] = ray.utils.get_num_cpus()

        if dtrain not in self._data:
            self.load_data(dtrain)

        local_dtrain = self._data[dtrain]

        local_evals = []
        for deval, name in evals:
            if deval not in self._data:
                self.load_data(deval)
            local_evals.append((self._data[deval], name))

        evals_result = dict()

        # Load model
        checkpoint_dir = os.path.dirname(self.checkpoint_file)
        if os.path.exists(self.checkpoint_file):
            kwargs.update({"xgb_model": self.checkpoint_file})
        elif not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir, 0o755, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Checkpoint directory {checkpoint_dir} could not be "
                    f"created."
                    f"\nFIX THIS by passing a valid (existing) location "
                    f"as the `checkpoint_path` parameter and omit using "
                    f"path separators (usually `/`) in the "
                    f"`checkpoint_prefix` parameter.") from e

        if "callbacks" in kwargs:
            callbacks = kwargs["callbacks"]
        else:
            callbacks = []
        callbacks.append(self._save_checkpoint_callback)
        kwargs["callbacks"] = callbacks

        with RabitContext(str(id(self)), rabit_args):
            bst = xgb.train(
                local_params,
                local_dtrain,
                *args,
                evals=local_evals,
                evals_result=evals_result,
                **kwargs)
            return {
                "bst": bst,
                "evals_result": evals_result,
                "train_n": self._local_n
            }

    def predict(self, model: xgb.Booster, data: RayDMatrix, **kwargs):
        _set_omp_num_threads()

        if data not in self._data:
            self.load_data(data)
        local_data = self._data[data]

        predictions = model.predict(local_data, **kwargs)
        return predictions


def _create_actor(rank: int,
                  num_actors: int,
                  num_cpus_per_actor: int,
                  num_gpus_per_actor: int,
                  resources_per_actor: Optional[Dict] = None,
                  queue: Optional[Queue] = None,
                  checkpoint_prefix: Optional[str] = None,
                  checkpoint_path: str = "/tmp",
                  checkpoint_frequency: int = 5) -> RayXGBoostActor:

    return RayXGBoostActor.options(
        num_cpus=num_cpus_per_actor,
        num_gpus=num_gpus_per_actor,
        resources=resources_per_actor).remote(
            rank=rank,
            num_actors=num_actors,
            queue=queue,
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_path=checkpoint_path,
            checkpoint_frequency=checkpoint_frequency)


def _trigger_data_load(actor, dtrain, evals):
    wait_load = [actor.load_data.remote(dtrain)]
    for deval, name in evals:
        wait_load.append(actor.load_data.remote(deval))
    return wait_load


def _cleanup(checkpoint_prefix: str, checkpoint_path: str, num_actors: int):
    for i in range(num_actors):
        checkpoint_file = _checkpoint_file(checkpoint_path, checkpoint_prefix,
                                           i)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)


def _shutdown(remote_workers: List[ActorHandle],
              queue: Optional[Queue] = None,
              force: bool = False):
    if force:
        logger.debug(f"Killing {len(remote_workers)} workers.")
        for worker in remote_workers:
            ray.kill(worker)
        if queue is not None:
            logger.debug("Killing Queue.")
            ray.kill(queue.actor)
    else:
        try:
            [worker.__ray_terminate__.remote() for worker in remote_workers]
            if queue is not None:
                queue.actor.__ray_terminate__.remote()
        except RayActorError:
            logger.warning("Failed to shutdown gracefully, forcing a "
                           "shutdown.")
            _shutdown(remote_workers, force=True)


def _train(params: Dict,
           dtrain: RayDMatrix,
           *args,
           evals=(),
           num_actors: int = 4,
           cpus_per_actor: int = 0,
           gpus_per_actor: int = -1,
           resources_per_actor: Optional[Dict] = None,
           checkpoint_prefix: Optional[str] = None,
           checkpoint_path: str = "/tmp",
           checkpoint_frequency: int = 5,
           **kwargs) -> Tuple[xgb.Booster, Dict, Dict]:
    _assert_ray_support()

    if not ray.is_initialized():
        ray.init()

    if gpus_per_actor == -1:
        gpus_per_actor = 0
        if "tree_method" in params and params["tree_method"].startswith("gpu"):
            gpus_per_actor = 1

    if cpus_per_actor <= 0:
        cluster_cpus = _ray_get_cluster_cpus() or 1
        cpus_per_actor = min(
            int(_get_max_node_cpus() or 1), int(cluster_cpus // num_actors))

    if "nthread" in params:
        if params["nthread"] > cpus_per_actor:
            raise ValueError(
                "Specified number of threads greater than number of CPUs. "
                "\nFIX THIS by passing a lower value for the `nthread` "
                "parameter or a higher number for `cpus_per_actor`.")
    else:
        params["nthread"] = cpus_per_actor

    # Create queue for communication from worker to caller.
    # Always create queue.
    queue = Queue()

    # Create remote actors
    actors = [
        _create_actor(i, num_actors, cpus_per_actor, gpus_per_actor,
                      resources_per_actor, queue, checkpoint_prefix,
                      checkpoint_path, checkpoint_frequency)
        for i in range(num_actors)
    ]
    logger.info(f"[RayXGBoost] Created {len(actors)} remote actors.")

    # Split data across workers
    wait_load = []
    for _, actor in enumerate(actors):
        wait_load.extend(_trigger_data_load(actor, dtrain, evals))

    try:
        ray.get(wait_load)
    except Exception:
        _shutdown(actors, queue, force=True)
        raise

    logger.info("[RayXGBoost] Starting XGBoost training.")

    # Start tracker
    env = _start_rabit_tracker(num_actors)
    rabit_args = [("%s=%s" % item).encode() for item in env.items()]

    # Train
    fut = [
        actor.train.remote(rabit_args, params, dtrain, evals, *args, **kwargs)
        for actor in actors
    ]

    callback_returns = [list() for _ in range(len(actors))]
    try:
        not_ready = fut
        while not_ready:
            if queue:
                while not queue.empty():
                    (actor_rank, item) = queue.get()
                    if isinstance(item, Callable):
                        item()
                    else:
                        callback_returns[actor_rank].append(item)
            ready, not_ready = ray.wait(not_ready, timeout=0)
            logger.debug("[RayXGBoost] Waiting for results...")
            ray.get(ready)
        # Once everything is ready
        ray.get(fut)
    # The inner loop should catch all exceptions
    except Exception:
        _shutdown(remote_workers=actors, queue=queue, force=True)
        raise

    # All results should be the same because of Rabit tracking. So we just
    # return the first one.
    res: Dict[str, Any] = ray.get(fut[0])
    bst = res["bst"]
    evals_result = res["evals_result"]
    additional_results = {}

    if callback_returns:
        additional_results["callback_returns"] = callback_returns

    all_res = ray.get(fut)
    total_n = sum(res["train_n"] or 0 for res in all_res)

    logger.info(f"[RayXGBoost] Finished XGBoost training on training data "
                f"with total N={total_n:,}.")

    if checkpoint_prefix:
        _cleanup(checkpoint_prefix, checkpoint_path, num_actors)

    _shutdown(remote_workers=actors, queue=queue, force=False)

    return bst, evals_result, additional_results


def train(params: Dict,
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
    """Distributed XGBoost training via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them train an
    XGBoost classifier. The XGBoost parameters will be shared and combined
    via Rabit's all-reduce protocol.

    If running inside a Ray Tune session, this function will automatically
    handle results to tune for hyperparameter search.

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
    max_actor_restarts = max_actor_restarts \
        if max_actor_restarts >= 0 else float("inf")
    _assert_ray_support()

    if not isinstance(dtrain, RayDMatrix):
        raise ValueError(
            "The `dtrain` argument passed to `train()` is not a RayDMatrix, "
            "but of type {}. "
            "\nFIX THIS by instantiating a RayDMatrix first: "
            "`dtrain = RayDMatrix(data=data, label=label)`.".format(
                type(dtrain)))

    _try_add_tune_callback(kwargs)

    if not dtrain.loaded and not dtrain.distributed:
        dtrain.load_data(num_actors)
    for (deval, name) in evals:
        if not deval.loaded and not deval.distributed:
            deval.load_data(num_actors)

    checkpoint_prefix = kwargs.pop("checkpoint_prefix",
                                   f".xgb_ray_{time.time()}")
    checkpoint_path = kwargs.pop("checkpoint_path", "/tmp")
    checkpoint_frequency = kwargs.pop("checkpoint_frequency", 5)

    bst = None
    train_evals_result = {}
    train_additional_results = {}

    tries = 0
    while tries <= max_actor_restarts:
        try:
            bst, train_evals_result, train_additional_results = _train(
                params,
                dtrain,
                *args,
                evals=evals,
                num_actors=num_actors,
                cpus_per_actor=cpus_per_actor,
                gpus_per_actor=gpus_per_actor,
                resources_per_actor=resources_per_actor,
                checkpoint_prefix=checkpoint_prefix,
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=checkpoint_frequency,
                **kwargs)
            break
        except RayActorError:
            if tries + 1 <= max_actor_restarts:
                logger.warning(
                    f"A Ray actor died during training. Trying to restart "
                    f"and continue training from last checkpoint "
                    f"(restart {tries + 1} of {max_actor_restarts}). "
                    f"Sleeping for 10 seconds for cleanup.")
                time.sleep(10)
            else:
                raise RuntimeError(
                    "A Ray actor died during training and the maximum number "
                    "of retries ({}) is exhausted. Checkpoints have been "
                    "stored at `{}` with prefix `{}` - you can pass these "
                    "parameters as `checkpoint_path` and `checkpoint_prefix` "
                    "to the `train()` function to try to continue "
                    "the training.".format(max_actor_restarts, checkpoint_path,
                                           checkpoint_prefix))
            tries += 1

    if isinstance(evals_result, dict):
        evals_result.update(train_evals_result)
    if isinstance(additional_results, dict):
        additional_results.update(train_additional_results)
    return bst


def _predict(model: xgb.Booster,
             data: RayDMatrix,
             num_actors: int = 4,
             cpus_per_actor: int = 0,
             gpus_per_actor: int = 0,
             resources_per_actor: Optional[Dict] = None,
             **kwargs):
    _assert_ray_support()

    if not ray.is_initialized():
        ray.init()

    # Create remote actors
    actors = [
        _create_actor(i, num_actors, cpus_per_actor, gpus_per_actor,
                      resources_per_actor) for i in range(num_actors)
    ]
    logger.info(f"[RayXGBoost] Created {len(actors)} remote actors.")

    # Split data across workers
    wait_load = []
    for _, actor in enumerate(actors):
        wait_load.extend(_trigger_data_load(actor, data, []))

    try:
        ray.get(wait_load)
    except Exception as exc:
        logger.warning(f"Caught an error during prediction: {str(exc)}")
        _shutdown(actors, force=True)
        raise

    # Put model into object store
    model_ref = ray.put(model)

    logger.info("[RayXGBoost] Starting XGBoost prediction.")

    # Train
    fut = [actor.predict.remote(model_ref, data, **kwargs) for actor in actors]

    try:
        actor_results = ray.get(fut)
    except Exception as exc:
        logger.warning(f"Caught an error during prediction: {str(exc)}")
        _shutdown(remote_workers=actors, force=True)
        raise

    _shutdown(remote_workers=actors, force=False)

    return combine_data(data.sharding, actor_results)


def predict(model: xgb.Booster,
            data: RayDMatrix,
            num_actors: int = 4,
            cpus_per_actor: int = 0,
            gpus_per_actor: int = 0,
            resources_per_actor: Optional[Dict] = None,
            max_actor_restarts: int = 0,
            **kwargs) -> Optional[np.ndarray]:
    """Distributed XGBoost predict via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them predict labels
    using an XGBoost booster model. The results are then combined and
    returned.

    Args:
        model (xgb.Booster): Booster object to call for prediction.
        data (RayDMatrix): Data object containing the prediction data.
        num_actors (int): Number of parallel Ray actors.
        cpus_per_actor (int): Number of CPUs to be used per Ray actor.
        gpus_per_actor (int): Number of GPUs to be used per Ray actor.
        resources_per_actor (Optional[Dict]): Dict of additional resources
            required per Ray actor.
        max_actor_restarts (int): Number of retries when Ray actors fail.
            Defaults to 0 (no retries). Set to -1 for unlimited retries.

    Returns: ``np.ndarray`` containing the predicted labels.

    """
    max_actor_restarts = max_actor_restarts \
        if max_actor_restarts >= 0 else float("inf")
    _assert_ray_support()

    if not isinstance(data, RayDMatrix):
        raise ValueError(
            "The `data` argument passed to `train()` is not a RayDMatrix, "
            "but of type {}. "
            "\nFIX THIS by instantiating a RayDMatrix first: "
            "`data = RayDMatrix(data=data)`.".format(type(data)))

    tries = 0
    while tries <= max_actor_restarts:
        try:
            return _predict(
                model,
                data,
                num_actors=num_actors,
                cpus_per_actor=cpus_per_actor,
                gpus_per_actor=gpus_per_actor,
                resources_per_actor=resources_per_actor,
                **kwargs)
        except RayActorError:
            if tries + 1 <= max_actor_restarts:
                logger.warning(
                    "A Ray actor died during prediction. Trying to restart "
                    "prediction from scratch. "
                    "Sleeping for 10 seconds for cleanup.")
                time.sleep(10)
            else:
                raise RuntimeError(
                    "A Ray actor died during prediction and the maximum "
                    "number of retries ({}) is exhausted.".format(
                        max_actor_restarts))
            tries += 1
    return None
