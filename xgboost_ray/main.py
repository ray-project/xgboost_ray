import threading
from typing import Tuple, Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass

import multiprocessing
import os
import pickle
import time

import numpy as np

import xgboost as xgb
from ray.util import placement_group
from ray.util.placement_group import PlacementGroup, remove_placement_group
from xgboost.core import XGBoostError

try:
    from xgboost.callback import TrainingCallback
except ImportError:
    print(f"xgboost_ray requires xgboost>=1.3 to work. Got version "
          f"{xgb.__version__}. Install latest release with "
          f"`pip install -U xgboost`.")

try:
    import ray
    from ray import logger
    from ray.services import get_node_ip_address
    from ray.exceptions import RayActorError, RayTaskError
    from ray.actor import ActorHandle
    from xgboost_ray.util import Event, Queue

    RAY_INSTALLED = True
except ImportError:
    ray = logger = get_node_ip_address = Queue = Event = ActorHandle = None
    RAY_INSTALLED = False

from xgboost_ray.tune import _try_add_tune_callback

from xgboost_ray.matrix import RayDMatrix, combine_data, \
    RayDeviceQuantileDMatrix, RayDataIter, concat_dataframes
from xgboost_ray.session import init_session, put_queue, \
    set_session_queue


class RayXGBoostTrainingError(RuntimeError):
    """Raised from RayXGBoostActor.train() when the local xgb.train function
    did not complete."""
    pass


class RayXGBoostTrainingStopped(RuntimeError):
    """Raised from RayXGBoostActor.train() when training was deliberately
    stopped."""
    pass


def _assert_ray_support():
    if not RAY_INSTALLED:
        raise ImportError(
            "Ray needs to be installed in order to use this module. "
            "Try: `pip install ray`")


def _start_rabit_tracker(num_workers: int):
    """Start Rabit tracker. The workers connect to this tracker to share
    their results.

    The Rabit tracker is the main process that all local workers connect to
    to share their weights. When one or more actors die, we want to
    restart the Rabit tracker, too, for two reasons: First we don't want to
    be potentially stuck with stale connections from old training processes.
    Second, we might restart training with a different number of actors, and
    for that we would have to restart the tracker anyway.

    To do this we start the Tracker in its own subprocess with its own PID.
    We can use this process then to specifically kill/terminate the tracker
    process in `_stop_rabit_tracker` without touching other functionality.
    """
    host = get_node_ip_address()

    env = {"DMLC_NUM_WORKER": num_workers}
    rabit_tracker = xgb.RabitTracker(hostIP=host, nslave=num_workers)

    # Get tracker Host + IP
    env.update(rabit_tracker.slave_envs())
    rabit_tracker.start(num_workers)

    # Wait until context completion
    process = multiprocessing.Process(target=rabit_tracker.join)
    process.daemon = True
    process.start()

    return process, env


def _stop_rabit_tracker(rabit_process):
    rabit_process.terminate()


class RabitContext:
    """This context is used by local training actors to connect to the
    Rabit tracker.

    Args:
        actor_id (str): Unique actor ID
        args (list): Arguments for Rabit initialisation. These are
            environment variables to configure Rabit clients.
    """

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


@dataclass
class RayParams:
    """Parameters to configure Ray-specific behavior.

    Args:
        num_actors (int): Number of parallel Ray actors.
        cpus_per_actor (int): Number of CPUs to be used per Ray actor.
        gpus_per_actor (int): Number of GPUs to be used per Ray actor.
        resources_per_actor (Optional[Dict]): Dict of additional resources
            required per Ray actor.
        elastic_training (bool): If True, training will continue with
            fewer actors if an actor fails. Default False.
        max_failed_actors (int): If `elastic_training` is True, this
            specifies the maximum number of failed actors with which
            we still continue training.
        max_actor_restarts (int): Number of retries when Ray actors fail.
            Defaults to 0 (no retries). Set to -1 for unlimited retries.
            Ignored when `elastic_training` is set.
        checkpoint_frequency (int): How often to save checkpoints. Defaults
            to ``5`` (every 5th iteration).
    """
    # Actor scheduling
    num_actors: int = 4
    cpus_per_actor: int = 0
    gpus_per_actor: int = -1
    resources_per_actor: Optional[Dict] = None

    # Fault tolerance
    elastic_training: bool = False
    max_failed_actors: int = 0
    max_actor_restarts: int = 0
    checkpoint_frequency: int = 5


@dataclass
class _Checkpoint:
    iteration: int = 0
    value: Optional[bytes] = None


def _validate_ray_params(ray_params: Union[None, RayParams, dict]) \
        -> RayParams:
    if ray_params is None:
        ray_params = RayParams()
    elif isinstance(ray_params, dict):
        ray_params = RayParams(**ray_params)
    elif not isinstance(ray_params, RayParams):
        raise ValueError(
            f"`ray_params` must be a `RayParams` instance, a dict, or None, "
            f"but it was {type(ray_params)}."
            f"\nFIX THIS preferably by passing a `RayParams` instance as "
            f"the `ray_params` parameter.")
    return ray_params


@ray.remote
class RayXGBoostActor:
    """Remote Ray XGBoost actor class.

    This remote actor handles local training and prediction of one data
    shard. It initializes a Rabit context, thus connecting to the Rabit
    all-reduce ring, and initializes local training, sending updates
    to other workers.

    The actor with rank 0 also checkpoints the model periodically and
    sends the checkpoint back to the driver.

    Args:
        rank (int): Rank of the actor. Must be ``0 <= rank < num_actors``.
        num_actors (int): Total number of actors.
        queue (Queue): Ray queue to communicate with main process.
        checkpoint_frequency (int): How often to store checkpoints. Defaults
            to ``5``, saving checkpoints every 5 boosting rounds.

    """

    def __init__(self,
                 rank: int,
                 num_actors: int,
                 queue: Optional[Queue] = None,
                 stop_event: Optional[Event] = None,
                 checkpoint_frequency: int = 5):
        self.queue = queue
        init_session(rank, self.queue)

        self.rank = rank
        self.num_actors = num_actors

        self.checkpoint_frequency = checkpoint_frequency

        self._data: Dict[RayDMatrix, xgb.DMatrix] = {}
        self._local_n = 0

        self._stop_event = stop_event

        _set_omp_num_threads()

    def set_queue(self, queue: Queue):
        self.queue = queue
        set_session_queue(self.queue)

    def set_stop_event(self, stop_event: Event):
        self._stop_event = stop_event

    def pid(self):
        """Get process PID. Used for checking if still alive"""
        return os.getpid()

    def _save_checkpoint_callback(self):
        """Send checkpoints to driver"""
        this = self

        class _SaveInternalCheckpointCallback(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if this.rank == 0 and \
                   epoch % this.checkpoint_frequency == 0:
                    put_queue(_Checkpoint(epoch, pickle.dumps(model)))

            def after_training(self, model):
                if this.rank == 0:
                    put_queue(_Checkpoint(-1, pickle.dumps(model)))
                return model

        return _SaveInternalCheckpointCallback()

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

        if "xgb_model" in kwargs:
            if isinstance(kwargs["xgb_model"], bytes):
                # bytearray type gets lost in remote actor call
                kwargs["xgb_model"] = bytearray(kwargs["xgb_model"])

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

        if "callbacks" in kwargs:
            callbacks = kwargs["callbacks"] or []
        else:
            callbacks = []
        callbacks.append(self._save_checkpoint_callback())
        kwargs["callbacks"] = callbacks

        result_dict = {}

        # We run xgb.train in a thread to be able to react to the stop event.
        def _train():
            try:
                with RabitContext(str(id(self)), rabit_args):
                    bst = xgb.train(
                        local_params,
                        local_dtrain,
                        *args,
                        evals=local_evals,
                        evals_result=evals_result,
                        **kwargs)
                    result_dict.update({
                        "bst": bst,
                        "evals_result": evals_result,
                        "train_n": self._local_n
                    })
            except XGBoostError:
                # Silent fail, will be raised as RayXGBoostTrainingStopped
                return

        thread = threading.Thread(target=_train)
        thread.daemon = True
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0)
            if self._stop_event.is_set():
                raise RayXGBoostTrainingStopped("Training was interrupted.")
            time.sleep(0.1)

        if not result_dict:
            raise RayXGBoostTrainingError("Training failed.")

        thread.join()
        return result_dict

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
                  placement_group: Optional[PlacementGroup] = None,
                  queue: Optional[Queue] = None,
                  checkpoint_frequency: int = 5) -> ActorHandle:

    return RayXGBoostActor.options(
        num_cpus=num_cpus_per_actor,
        num_gpus=num_gpus_per_actor,
        resources=resources_per_actor,
        placement_group=placement_group).remote(
            rank=rank,
            num_actors=num_actors,
            queue=queue,
            checkpoint_frequency=checkpoint_frequency)


def _trigger_data_load(actor, dtrain, evals):
    wait_load = [actor.load_data.remote(dtrain)]
    for deval, name in evals:
        wait_load.append(actor.load_data.remote(deval))
    return wait_load


def _handle_queue(queue: Queue, checkpoint: _Checkpoint,
                  callback_returns: Dict):
    """Handle results obtained from workers through the remote Queue object.

    Remote actors supply these results via the
    ``xgboost_ray.session.put_queue()`` function. These can be:

    - Callables. These will be called immediately with no arguments.
    - ``_Checkpoint`` objects. These will update the latest checkpoint
      object on the driver.
    - Any other type. These will be appended to an actor rank-specific
      ``callback_returns`` dict that will be written to the
      ``additional_returns`` dict of the :func:`train() <train>` method.
    """
    while not queue.empty():
        (actor_rank, item) = queue.get()
        if isinstance(item, Callable):
            item()
        elif isinstance(item, _Checkpoint):
            checkpoint.__dict__.update(item.__dict__)
        else:
            callback_returns[actor_rank].append(item)


def _get_actor_alive_status(actors: List[ActorHandle],
                            callback: Callable[[ActorHandle], None]):
    """Loop through all actors. Invoke a callback on dead actors. """
    obj_to_rank = {}

    alive = 0
    dead = 0

    for rank, actor in enumerate(actors):
        if actor is None:
            dead += 1
            continue
        obj = actor.pid.remote()
        obj_to_rank[obj] = rank

    not_ready = list(obj_to_rank.keys())
    while not_ready:
        ready, not_ready = ray.wait(not_ready, timeout=0)

        for obj in ready:
            try:
                pid = ray.get(obj)
                rank = obj_to_rank[obj]
                logger.debug(
                    f"Actor {actors[rank]} with PID {pid} is still alive.")
                alive += 1
            except Exception:
                rank = obj_to_rank[obj]
                logger.debug(f"Actor {actors[rank]} is _not_ alive.")
                dead += 1
                callback(actors[rank])
    logger.info(f"Actor status: {alive} alive, {dead} dead "
                f"({alive+dead} total)")

    return alive, dead


def _shutdown(actors: List[ActorHandle],
              queue: Optional[Queue] = None,
              event: Optional[Event] = None,
              placement_group: Optional[PlacementGroup] = None,
              force: bool = False):
    for i in range(len(actors)):
        actor = actors[i]
        if actor is None:
            continue
        if force:
            ray.kill(actor)
        else:
            try:
                ray.get(actor.__ray_terminate__.remote())
            except RayActorError:
                ray.kill(actor)
        actors[i] = None
    if queue:
        queue.shutdown()
    if event:
        event.shutdown()
    if placement_group:
        remove_placement_group(placement_group)

def _create_communication_processes():
    # Create Queue and Event actors and make sure to colocate with driver node.
    node_ip = ray.services.get_node_ip_address()
    # Have to explicitly set num_cpus to 0.
    placement_option = {"num_cpus": 0, "resources": {f"node:{node_ip}": 0.01}}
    queue = Queue(actor_options=placement_option)  # Queue actor
    stop_event = Event(actor_options=placement_option)  # Stop event actor
    return queue, stop_event

def _create_placement_group(cpus_per_actor, gpus_per_actor,
                            resources_per_actor, num_actors):
    resources_per_bundle = {"CPU": cpus_per_actor, "GPU": gpus_per_actor}
    extra_resources_per_bundle = {} if resources_per_actor is None else \
        resources_per_actor
    # Create placement group for training worker colocation.
    bundles = [{**resources_per_bundle, **extra_resources_per_bundle} for _ in range(num_actors)]
    pg = placement_group(bundles, strategy="PACK")
    # Wait for placement group to get created.
    logger.debug("Waiting for placement group to start.")
    ready = ray.wait([pg.ready()], timeout=100)
    if ready:
        logger.debug("Placement group has started.")
    else:
        raise TimeoutError("Placement group creation timed out. Make sure "
                           "your cluster either has enough resources or use "
                           "an autoscaling cluster.")
    return pg


def _train(params: Dict,
           dtrain: RayDMatrix,
           *args,
           evals=(),
           ray_params: RayParams,
           cpus_per_actor: int,
           gpus_per_actor: int,
           _checkpoint: _Checkpoint,
           _additional_results: Dict,
           _actors: List,
           _queue: Queue,
           _stop_event: Event,
           _placement_group: PlacementGroup,
           _failed_actor_ranks: set,
           **kwargs) -> Tuple[xgb.Booster, Dict, Dict]:
    """This is the local train function wrapped by :func:`train() <train>`.

    This function can be thought of one invocation of a multi-actor xgboost
    training run. It starts the required number of actors, triggers data
    loading, collects the results, and handles (i.e. registers) actor failures
    - but it does not handle fault tolerance or general training setup.

    Generally, this function is called one or multiple times by the
    :func:`train() <train>` function. It is called exactly once if no
    errors occur. It is called more than once if errors occurred (e.g. an
    actor died) and failure handling is enabled.
    """
    if "nthread" in params:
        if params["nthread"] > cpus_per_actor:
            raise ValueError(
                "Specified number of threads greater than number of CPUs. "
                "\nFIX THIS by passing a lower value for the `nthread` "
                "parameter or a higher number for `cpus_per_actor`.")
    else:
        params["nthread"] = cpus_per_actor

    # This is a callback that handles actor failures.
    # We identify the rank of the failed actor, add this to a set of
    # failed actors (which we might want to restart later), and set its
    # entry in the actor list to None.
    def handle_actor_failure(actor_id):
        rank = _actors.index(actor_id)
        _failed_actor_ranks.add(rank)
        _actors[rank] = None

    # Here we create new actors. In the first invocation of _train(), this
    # will be all actors. In future invocations, this may be less than
    # the num_actors setting, depending on the failure mode.
    newly_created = 0
    for i in list(_failed_actor_ranks):
        if _actors[i] is not None:
            raise RuntimeError(
                f"Trying to create actor with rank {i}, but it already "
                f"exists.")
        actor = _create_actor(
            rank=i,
            num_actors=ray_params.num_actors,
            num_cpus_per_actor=cpus_per_actor,
            num_gpus_per_actor=gpus_per_actor,
            resources_per_actor=ray_params.resources_per_actor,
            placement_group=_placement_group,
            queue=_queue,
            checkpoint_frequency=ray_params.checkpoint_frequency)
        # Set actor entry in our list
        _actors[i] = actor
        # Remove from this set so it is not created again
        _failed_actor_ranks.remove(i)
        newly_created += 1

    # Maybe we got a new Queue actor, so send it to all actors.
    wait_queue = [
        actor.set_queue.remote(_queue) for actor in _actors
        if actor is not None
    ]
    ray.get(wait_queue)

    # Maybe we got a new Event actor, so send it to all actors.
    wait_event = [
        actor.set_stop_event.remote(_stop_event) for actor in _actors
        if actor is not None
    ]
    ray.get(wait_event)

    alive_actors = sum(1 for a in _actors if a is not None)
    logger.info(f"[RayXGBoost] Created {newly_created} new actors "
                f"({alive_actors} total actors).")

    # Split data across workers
    wait_load = []
    for actor in _actors:
        if actor is None:
            continue
        # If data is already on the node, will not load again
        wait_load.extend(_trigger_data_load(actor, dtrain, evals))

    try:
        ray.get(wait_load)
    except Exception as exc:
        _stop_event.set()
        _get_actor_alive_status(_actors, handle_actor_failure)
        raise RayActorError from exc

    logger.info("[RayXGBoost] Starting XGBoost training.")

    # Start Rabit tracker for gradient sharing
    rabit_process, env = _start_rabit_tracker(alive_actors)
    rabit_args = [("%s=%s" % item).encode() for item in env.items()]

    # Load checkpoint if we have one. In that case we need to adjust the
    # number of training rounds.
    if _checkpoint.value:
        kwargs["xgb_model"] = pickle.loads(_checkpoint.value)
        if _checkpoint.iteration == -1:
            # -1 means training already finished.
            logger.error(
                f"Trying to load continue from checkpoint, but the checkpoint"
                f"indicates training already finished. Returning last"
                f"checkpointed model instead.")
            return kwargs["xgb_model"], {}, _additional_results

        kwargs["num_boost_round"] = kwargs.get("num_boost_round", 10) - \
            _checkpoint.iteration - 1

    # The callback_returns dict contains actor-rank indexed lists of
    # results obtained through the `put_queue` function, usually
    # sent via callbacks.
    callback_returns = _additional_results.get("callback_returns")
    if callback_returns is None:
        callback_returns = [list() for _ in range(len(_actors))]
        _additional_results["callback_returns"] = callback_returns

    # Trigger the train function
    training_futures = [
        actor.train.remote(rabit_args, params, dtrain, evals, *args, **kwargs)
        for actor in _actors if actor is not None
    ]

    # Failure handling loop. Here we wait until all training tasks finished.
    # If a training task fails, we stop training on the remaining actors,
    # check which ones are still alive, and raise the error.
    # The train() wrapper function will then handle the error.
    try:
        not_ready = training_futures
        while not_ready:
            if _queue:
                _handle_queue(
                    queue=_queue,
                    checkpoint=_checkpoint,
                    callback_returns=callback_returns)
            ready, not_ready = ray.wait(not_ready, timeout=0)
            logger.debug("[RayXGBoost] Waiting for results...")
            ray.get(ready)
        # Once everything is ready
        ray.get(training_futures)

        # Get items from queue one last time
        if _queue:
            _handle_queue(
                queue=_queue,
                checkpoint=_checkpoint,
                callback_returns=callback_returns)

    # The inner loop should catch all exceptions
    except Exception as exc:
        # Stop all other actors from training
        _stop_event.set()

        # Check which actors are still alive
        _get_actor_alive_status(_actors, handle_actor_failure)

        # Todo: Try to fetch newer checkpoint, store in `_checkpoint`
        # Shut down rabit
        _stop_rabit_tracker(rabit_process)

        raise RayActorError from exc

    # Training is now complete.
    # Stop Rabit tracking process
    _stop_rabit_tracker(rabit_process)

    # Get all results from all actors.
    all_results: List[Dict[str, Any]] = ray.get(training_futures)

    # All results should be the same because of Rabit tracking. So we just
    # return the first one.
    bst = all_results[0]["bst"]
    evals_result = all_results[0]["evals_result"]

    if callback_returns:
        _additional_results["callback_returns"] = callback_returns

    total_n = sum(res["train_n"] or 0 for res in all_results)

    _additional_results["total_n"] = total_n

    logger.info(f"[RayXGBoost] Finished XGBoost training on training data "
                f"with total N={total_n:,}.")

    return bst, evals_result, _additional_results

def train(params: Dict,
          dtrain: RayDMatrix,
          *args,
          evals=(),
          evals_result: Optional[Dict] = None,
          additional_results: Optional[Dict] = None,
          ray_params: Union[None, RayParams, Dict] = None,
          **kwargs):
    """Distributed XGBoost training via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them train an
    XGBoost classifier. The XGBoost parameters will be shared and combined
    via Rabit's all-reduce protocol.

    If running inside a Ray Tune session, this function will automatically
    handle results to tune for hyperparameter search.

    Failure handling:

    XGBoost on Ray supports automatic failure handling that can be configured
    with the :class:`ray_params <RayParams>` argument. If an actor or local
    training task dies, the Ray actor is marked as dead, and there are
    three options on how to proceed.

    First, if ``ray_params.elastic_training`` is ``True`` and
    the number of dead actors is below ``ray_params.max_failed_actors``,
    training will continue right away with fewer actors. No data will be
    loaded again and the latest available checkpoint will be used.

    Second, if ``ray_params.elastic_training`` is ``False`` and
    the number of restarts is below ``ray_params.max_actor_restarts``,
    Ray will try to schedule the dead actor again, load the data shard
    on this actor, and then continue training from the latest checkpoint.

    Third, if none of the above is the case, training is aborted.

    Args:
        params (Dict): parameter dict passed to ``xgboost.train()``
        dtrain (RayDMatrix): Data object containing the training data.
        evals (Union[List[Tuple], Tuple]): ``evals`` tuple passed to
            ``xgboost.train()``.
        evals_result (Optional[Dict]): Dict to store evaluation results in.
        additional_results (Optional[Dict]): Dict to store additional results.
        ray_params (Union[None, RayParams, Dict]): Parameters to configure
            Ray-specific behavior. See :class:`RayParams` for a list of valid
            configuration parameters.
        **kwargs: Keyword arguments will be passed to the local
            `xgb.train()` calls.

    Returns: An ``xgboost.Booster`` object.
    """
    ray_params = _validate_ray_params(ray_params)

    max_actor_restarts = ray_params.max_actor_restarts \
        if ray_params.max_actor_restarts >= 0 else float("inf")
    _assert_ray_support()

    if not isinstance(dtrain, RayDMatrix):
        raise ValueError(
            "The `dtrain` argument passed to `train()` is not a RayDMatrix, "
            "but of type {}. "
            "\nFIX THIS by instantiating a RayDMatrix first: "
            "`dtrain = RayDMatrix(data=data, label=label)`.".format(
                type(dtrain)))

    if not ray.is_initialized():
        ray.init()

    gpus_per_actor = ray_params.gpus_per_actor
    cpus_per_actor = ray_params.cpus_per_actor

    # Automatically set gpus_per_actor if left at the default value
    if gpus_per_actor == -1:
        gpus_per_actor = 0
        if "tree_method" in params and params["tree_method"].startswith("gpu"):
            gpus_per_actor = 1

    # Automatically set cpus_per_actor if left at the default value
    # Will be set to the number of cluster CPUs divided by the number of
    # actors, bounded by the maximum number of CPUs across actors nodes.
    if cpus_per_actor <= 0:
        cluster_cpus = _ray_get_cluster_cpus() or 1
        cpus_per_actor = min(
            int(_get_max_node_cpus() or 1),
            int(cluster_cpus // ray_params.num_actors))

    _try_add_tune_callback(kwargs)

    if not dtrain.loaded and not dtrain.distributed:
        dtrain.load_data(ray_params.num_actors)
    for (deval, name) in evals:
        if not deval.loaded and not deval.distributed:
            deval.load_data(ray_params.num_actors)

    bst = None
    train_evals_result = {}
    train_additional_results = {}

    tries = 0
    checkpoint = _Checkpoint()  # Keep track of latest checkpoint
    current_results = {}  # Keep track of additional results
    actors = [None] * ray_params.num_actors  # All active actors

    # Create the Queue and Event actors.
    queue, stop_event = _create_communication_processes()

    pg = _create_placement_group(cpus_per_actor, gpus_per_actor,
                                 ray_params.resources_per_actor,
                                 ray_params.num_actors)

    start_actor_ranks = set(range(ray_params.num_actors))  # Start these
    while tries <= max_actor_restarts:
        try:
            bst, train_evals_result, train_additional_results = _train(
                params,
                dtrain,
                *args,
                evals=evals,
                ray_params=ray_params,
                cpus_per_actor=cpus_per_actor,
                gpus_per_actor=gpus_per_actor,
                _checkpoint=checkpoint,
                _additional_results=current_results,
                _actors=actors,
                _queue=queue,
                _stop_event=stop_event,
                _placement_group=pg,
                _failed_actor_ranks=start_actor_ranks,
                **kwargs)
            break
        except (RayActorError, RayTaskError) as exc:
            alive_actors = sum(1 for a in actors if a is not None)
            start_again = False
            if ray_params.elastic_training:
                if alive_actors < ray_params.num_actors - \
                   ray_params.max_failed_actors:
                    raise RuntimeError(
                        "A Ray actor died during training and the maximum "
                        "number of dead actors in elastic training was "
                        "reached. Shutting down training.") from exc
                # Do not start new actors
                start_actor_ranks.clear()
                logger.warning(
                    f"A Ray actor died during training. Trying to continue "
                    f"training on the remaining actors. "
                    f"This will use {alive_actors} existing actors and start "
                    f"{len(start_actor_ranks)} new actors. "
                    f"Sleeping for 10 seconds for cleanup.")
                start_again = True

            if tries + 1 <= max_actor_restarts:
                logger.warning(
                    f"A Ray actor died during training. Trying to restart "
                    f"and continue training from last checkpoint "
                    f"(restart {tries + 1} of {max_actor_restarts}). "
                    f"This will use {alive_actors} existing actors and start "
                    f"{len(start_actor_ranks)} new actors. "
                    f"Sleeping for 10 seconds for cleanup.")
                start_again = True

            if start_again:
                time.sleep(5)
                queue.shutdown()
                stop_event.shutdown()
                #remove_placement_group(pg)
                time.sleep(5)
                queue, stop_event = _create_communication_processes()
                # pg = _create_placement_group(cpus_per_actor, gpus_per_actor,
                #                              resources_per_actor=ray_params.resources_per_actor, num_actors=alive_actors+len(start_actor_ranks))
            else:
                raise RuntimeError(
                    f"A Ray actor died during training and the maximum number "
                    f"of retries ({max_actor_restarts}) is exhausted."
                ) from exc
            tries += 1

    _shutdown(actors=actors, queue=queue, event=stop_event,
              placement_group=pg, force=False)

    if isinstance(evals_result, dict):
        evals_result.update(train_evals_result)
    if isinstance(additional_results, dict):
        additional_results.update(train_additional_results)
    return bst


def _predict(model: xgb.Booster, data: RayDMatrix, ray_params: RayParams,
             **kwargs):
    _assert_ray_support()

    if not ray.is_initialized():
        ray.init()

    # Create remote actors
    actors = [
        _create_actor(
            rank=i,
            num_actors=ray_params.num_actors,
            num_cpus_per_actor=ray_params.cpus_per_actor,
            num_gpus_per_actor=ray_params.gpus_per_actor
            if ray_params.gpus_per_actor >= 0 else 0,
            resources_per_actor=ray_params.resources_per_actor)
        for i in range(ray_params.num_actors)
    ]
    logger.info(f"[RayXGBoost] Created {len(actors)} remote actors.")

    # Split data across workers
    wait_load = []
    for actor in actors:
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
        _shutdown(actors=actors, force=True)
        raise

    _shutdown(actors=actors, force=False)

    return combine_data(data.sharding, actor_results)


def predict(model: xgb.Booster,
            data: RayDMatrix,
            ray_params: Union[None, RayParams, Dict] = None,
            **kwargs) -> Optional[np.ndarray]:
    """Distributed XGBoost predict via Ray.

    This function will connect to a Ray cluster, create ``num_actors``
    remote actors, send data shards to them, and have them predict labels
    using an XGBoost booster model. The results are then combined and
    returned.

    Args:
        model (xgb.Booster): Booster object to call for prediction.
        data (RayDMatrix): Data object containing the prediction data.
        ray_params (Union[None, RayParams, Dict]): Parameters to configure
            Ray-specific behavior. See :class:`RayParams` for a list of valid
            configuration parameters.
        **kwargs: Keyword arguments will be passed to the local
            `xgb.predict()` calls.

    Returns: ``np.ndarray`` containing the predicted labels.

    """
    ray_params = _validate_ray_params(ray_params)

    max_actor_restarts = ray_params.max_actor_restarts \
        if ray_params.max_actor_restarts >= 0 else float("inf")
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
            return _predict(model, data, ray_params=ray_params, **kwargs)
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
