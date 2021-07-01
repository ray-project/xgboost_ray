from typing import Tuple, Dict, Any, List, Optional, Callable, Union, Sequence
from dataclasses import dataclass, field
from distutils.version import LooseVersion

import multiprocessing
import os
import pickle
import time
import threading
import warnings
import re

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost.core import XGBoostError, EarlyStopException

from xgboost_ray.callback import DistributedCallback, \
    DistributedCallbackContainer
from xgboost_ray.compat import TrainingCallback, RabitTracker, LEGACY_CALLBACK

try:
    import ray
    from ray import logger
    from ray.services import get_node_ip_address
    from ray.exceptions import RayActorError, RayTaskError
    from ray.actor import ActorHandle
    from ray.util import placement_group
    from ray.util.placement_group import PlacementGroup, \
        remove_placement_group, get_current_placement_group

    from xgboost_ray.util import Event, Queue, MultiActorTask, \
        force_on_current_node

    if LooseVersion(ray.__version__) >= LooseVersion("1.5.0"):
        # https://github.com/ray-project/ray/pull/16437
        DEFAULT_PG = "default"
    else:
        DEFAULT_PG = None

    RAY_INSTALLED = True
except ImportError:
    ray = get_node_ip_address = Queue = Event = ActorHandle = logger = None
    RAY_INSTALLED = False

from xgboost_ray.tune import _try_add_tune_callback, _get_tune_resources, \
    TUNE_USING_PG, is_session_enabled

from xgboost_ray.matrix import RayDMatrix, combine_data, \
    RayDeviceQuantileDMatrix, RayDataIter, concat_dataframes, \
    LEGACY_MATRIX
from xgboost_ray.session import init_session, put_queue, \
    set_session_queue

# Whether to use SPREAD placement group strategy for training.
_USE_SPREAD_STRATEGY = int(os.getenv("RXGB_USE_SPREAD_STRATEGY", 1))

# How long to wait for placement group creation before failing.
PLACEMENT_GROUP_TIMEOUT_S = int(
    os.getenv("RXGB_PLACEMENT_GROUP_TIMEOUT_S", 100))

# Status report frequency when waiting for initial actors and during training
STATUS_FREQUENCY_S = int(os.getenv("RXGB_STATUS_FREQUENCY_S", 30))

# If restarting failed actors is disabled
ELASTIC_RESTART_DISABLED = bool(
    int(os.getenv("RXGB_ELASTIC_RESTART_DISABLED", 0)))

# How often to check for new available resources
ELASTIC_RESTART_RESOURCE_CHECK_S = int(
    os.getenv("RXGB_ELASTIC_RESTART_RESOURCE_CHECK_S", 30))

# How long to wait before triggering a new start of the training loop
# when new actors become available
ELASTIC_RESTART_GRACE_PERIOD_S = int(
    os.getenv("RXGB_ELASTIC_RESTART_GRACE_PERIOD_S", 10))

LEGACY_WARNING = (
    f"You are using `xgboost_ray` with a legacy XGBoost version "
    f"(version {xgb.__version__}). While we try to support "
    f"older XGBoost versions, please note that this library is only "
    f"fully tested and supported for XGBoost >= 1.4. Please consider "
    f"upgrading your XGBoost version (`pip install -U xgboost`).")

# XGBoost version as an int tuple for comparisions
XGBOOST_VERSION_TUPLE = tuple(
    [int(x) for x in re.sub(r"[^\.0-9]", "", xgb.__version__).split(".")])


class RayXGBoostTrainingError(RuntimeError):
    """Raised from RayXGBoostActor.train() when the local xgb.train function
    did not complete."""
    pass


class RayXGBoostTrainingStopped(RuntimeError):
    """Raised from RayXGBoostActor.train() when training was deliberately
    stopped."""
    pass


class RayXGBoostActorAvailable(RuntimeError):
    """Raise from `_update_scheduled_actor_states()` when new actors become
    available in elastic training"""
    pass


def _assert_ray_support():
    if not RAY_INSTALLED:
        raise ImportError(
            "Ray needs to be installed in order to use this module. "
            "Try: `pip install ray`")


def _maybe_print_legacy_warning():
    if LEGACY_MATRIX or LEGACY_CALLBACK:
        warnings.warn(LEGACY_WARNING)


def _is_client_connected() -> bool:
    try:
        return ray.util.client.ray.is_connected()
    except Exception:
        return False


class _RabitTracker(RabitTracker):
    """
    This method overwrites the xgboost-provided RabitTracker to switch
    from a daemon thread to a multiprocessing Process. This is so that
    we are able to terminate/kill the tracking process at will.
    """

    def start(self, nslave):
        # TODO: refactor RabitTracker to support spawn process creation.
        # In python 3.8, spawn is used as default process creation on macOS.
        # But spawn doesn't work because `run` is not pickleable.
        # For now we force the start method to use fork.
        multiprocessing.set_start_method("fork", force=True)

        def run():
            self.accept_slaves(nslave)

        self.thread = multiprocessing.Process(target=run, args=())
        self.thread.start()


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

    rabit_tracker = _RabitTracker(hostIP=host, nslave=num_workers)

    # Get tracker Host + IP
    env.update(rabit_tracker.slave_envs())
    rabit_tracker.start(num_workers)

    logger.debug(
        f"Started Rabit tracker process with PID {rabit_tracker.thread.pid}")

    return rabit_tracker.thread, env


def _stop_rabit_tracker(rabit_process: multiprocessing.Process):
    logger.debug(f"Stopping Rabit process with PID {rabit_process.pid}")
    rabit_process.join(timeout=5)
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
    resource_ids = ray.worker.get_resource_ids()
    if "CPU" in resource_ids:
        return sum(cpu[1] for cpu in resource_ids["CPU"])
    return None


def _ray_get_cluster_cpus():
    return ray.cluster_resources().get("CPU", None)


def _get_min_node_cpus():
    max_node_cpus = min(
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
    if not LEGACY_MATRIX and isinstance(data, RayDeviceQuantileDMatrix):
        # If we only got a single data shard, create a list so we can
        # iterate over it
        if not isinstance(param["data"], list):
            param["data"] = [param["data"]]

            if not isinstance(param["label"], list):
                param["label"] = [param["label"]]
            if not isinstance(param["weight"], list):
                param["weight"] = [param["weight"]]
            if not isinstance(param["data"], list):
                param["base_margin"] = [param["base_margin"]]

        param["label_lower_bound"] = [None]
        param["label_upper_bound"] = [None]

        dm_param = {
            "feature_names": data.feature_names,
            "feature_types": data.feature_types,
            "missing": data.missing,
        }
        param.update(dm_param)
        it = RayDataIter(**param)
        matrix = xgb.DeviceQuantileDMatrix(it, **dm_param)
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

        if LEGACY_MATRIX:
            param.pop("base_margin", None)

        matrix = xgb.DMatrix(**param)

        if not LEGACY_MATRIX:
            matrix.set_info(label_lower_bound=ll, label_upper_bound=lu)

    data.update_matrix_properties(matrix)
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

    # Distributed callbacks
    distributed_callbacks: Optional[List[DistributedCallback]] = None

    def get_tune_resources(self):
        """Return the resources to use for xgboost_ray training with Tune."""
        if self.cpus_per_actor <= 0 or self.num_actors <= 0:
            raise ValueError("num_actors and cpus_per_actor both must be "
                             "greater than 0.")
        return _get_tune_resources(
            num_actors=self.num_actors,
            cpus_per_actor=self.cpus_per_actor,
            gpus_per_actor=max(0, self.gpus_per_actor),
            resources_per_actor=self.resources_per_actor)


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
    if ray_params.num_actors < 2:
        warnings.warn(
            f"`num_actors` in `ray_params` is smaller than 2 "
            f"({ray_params.num_actors}). XGBoost will NOT be distributed!")
    return ray_params


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

    def __init__(
            self,
            rank: int,
            num_actors: int,
            queue: Optional[Queue] = None,
            stop_event: Optional[Event] = None,
            checkpoint_frequency: int = 5,
            distributed_callbacks: Optional[List[DistributedCallback]] = None):
        self.queue = queue
        init_session(rank, self.queue)

        self.rank = rank
        self.num_actors = num_actors

        self.checkpoint_frequency = checkpoint_frequency

        self._data: Dict[RayDMatrix, xgb.DMatrix] = {}
        self._local_n: Dict[RayDMatrix, int] = {}

        self._stop_event = stop_event

        self._distributed_callbacks = DistributedCallbackContainer(
            distributed_callbacks)

        self._distributed_callbacks.on_init(self)
        _set_omp_num_threads()
        logger.debug(f"Initialized remote XGBoost actor with rank {self.rank}")

    def set_queue(self, queue: Queue):
        self.queue = queue
        set_session_queue(self.queue)

    def set_stop_event(self, stop_event: Event):
        self._stop_event = stop_event

    def _get_stop_event(self):
        return self._stop_event

    def pid(self):
        """Get process PID. Used for checking if still alive"""
        return os.getpid()

    def ip(self):
        """Get node IP address."""
        return get_node_ip_address()

    def _save_checkpoint_callback(self):
        """Send checkpoints to driver"""
        this = self

        class _SaveInternalCheckpointCallback(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if xgb.rabit.get_rank() == 0 and \
                        epoch % this.checkpoint_frequency == 0:
                    put_queue(_Checkpoint(epoch, pickle.dumps(model)))

            def after_training(self, model):
                if xgb.rabit.get_rank() == 0:
                    put_queue(_Checkpoint(-1, pickle.dumps(model)))
                return model

        return _SaveInternalCheckpointCallback()

    def _stop_callback(self):
        """Stop if event is set"""
        this = self
        # Keep track of initial stop event. Since we're training in a thread,
        # the stop event might be overwritten, which should he handled
        # as if the previous stop event was set.
        initial_stop_event = self._stop_event

        class _StopCallback(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                try:
                    if this._stop_event.is_set() or \
                            this._get_stop_event() is not initial_stop_event:
                        if LEGACY_CALLBACK:
                            raise EarlyStopException(epoch)
                        # Returning True stops training
                        return True
                except RayActorError:
                    if LEGACY_CALLBACK:
                        raise EarlyStopException(epoch)
                    return True

        return _StopCallback()

    def load_data(self, data: RayDMatrix):
        if data in self._data:
            return

        self._distributed_callbacks.before_data_loading(self, data)

        param = data.get_data(self.rank, self.num_actors)
        if isinstance(param["data"], list):
            self._local_n[data] = sum(len(a) for a in param["data"])
        else:
            self._local_n[data] = len(param["data"])
        data.unload_data()  # Free object store

        matrix = _get_dmatrix(data, param)
        self._data[data] = matrix

        self._distributed_callbacks.after_data_loading(self, data)

    def train(self, rabit_args: List[str], return_bst: bool,
              params: Dict[str, Any], dtrain: RayDMatrix,
              evals: Tuple[RayDMatrix, str], *args,
              **kwargs) -> Dict[str, Any]:
        self._distributed_callbacks.before_train(self)

        num_threads = _set_omp_num_threads()

        local_params = params.copy()

        if "xgb_model" in kwargs:
            if isinstance(kwargs["xgb_model"], bytes):
                # bytearray type gets lost in remote actor call
                kwargs["xgb_model"] = bytearray(kwargs["xgb_model"])

        if "nthread" not in local_params and "n_jobs" not in local_params:
            if num_threads > 0:
                local_params["nthread"] = num_threads
                local_params["n_jobs"] = num_threads
            else:
                local_params["nthread"] = sum(
                    num
                    for _, num in ray.worker.get_resource_ids().get("CPU", []))
                local_params["n_jobs"] = local_params["nthread"]

        if dtrain not in self._data:
            self.load_data(dtrain)

        local_dtrain = self._data[dtrain]

        if not local_dtrain.get_label().size:
            raise RuntimeError(
                "Training data has no label set. Please make sure to set "
                "the `label` argument when initializing `RayDMatrix()` "
                "for data you would like to train on.")

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
        callbacks.append(self._stop_callback())
        kwargs["callbacks"] = callbacks

        result_dict = {}
        error_dict = {}

        # We run xgb.train in a thread to be able to react to the stop event.
        def _train():
            try:
                with RabitContext(str(id(self)), rabit_args):
                    if LEGACY_CALLBACK:
                        for xgb_callback in kwargs.get("callbacks", []):
                            if isinstance(xgb_callback, TrainingCallback):
                                xgb_callback.before_training(None)

                    bst = xgb.train(
                        local_params,
                        local_dtrain,
                        *args,
                        evals=local_evals,
                        evals_result=evals_result,
                        **kwargs)

                    if LEGACY_CALLBACK:
                        for xgb_callback in kwargs.get("callbacks", []):
                            if isinstance(xgb_callback, TrainingCallback):
                                xgb_callback.after_training(bst)

                    result_dict.update({
                        "bst": bst,
                        "evals_result": evals_result,
                        "train_n": self._local_n[dtrain]
                    })
            except EarlyStopException:
                # Usually this should be caught by XGBoost core.
                # Silent fail, will be raised as RayXGBoostTrainingStopped.
                return
            except XGBoostError as e:
                error_dict.update({"exception": e})
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
            raise_from = error_dict.get("exception", None)
            raise RayXGBoostTrainingError("Training failed.") from raise_from

        thread.join()
        self._distributed_callbacks.after_train(self, result_dict)

        if not return_bst:
            result_dict.pop("bst", None)

        return result_dict

    def predict(self, model: xgb.Booster, data: RayDMatrix, **kwargs):
        self._distributed_callbacks.before_predict(self)

        _set_omp_num_threads()

        if data not in self._data:
            self.load_data(data)
        local_data = self._data[data]

        predictions = model.predict(local_data, **kwargs)
        if predictions.ndim == 1:
            callback_predictions = pd.Series(predictions)
        else:
            callback_predictions = pd.DataFrame(predictions)
        self._distributed_callbacks.after_predict(self, callback_predictions)
        return predictions


@ray.remote
class _RemoteRayXGBoostActor(RayXGBoostActor):
    pass


class _PrepareActorTask(MultiActorTask):
    def __init__(self, actor: ActorHandle, queue: Queue, stop_event: Event,
                 load_data: List[RayDMatrix]):
        futures = []
        futures.append(actor.set_queue.remote(queue))
        futures.append(actor.set_stop_event.remote(stop_event))
        for data in load_data:
            futures.append(actor.load_data.remote(data))

        super(_PrepareActorTask, self).__init__(futures)


def _autodetect_resources(ray_params: RayParams,
                          use_tree_method: bool = False) -> Tuple[int, int]:
    gpus_per_actor = ray_params.gpus_per_actor
    cpus_per_actor = ray_params.cpus_per_actor

    # Automatically set gpus_per_actor if left at the default value
    if gpus_per_actor == -1:
        gpus_per_actor = 0
        if use_tree_method:
            gpus_per_actor = 1

    # Automatically set cpus_per_actor if left at the default value
    # Will be set to the number of cluster CPUs divided by the number of
    # actors, bounded by the minimum number of CPUs across actors nodes.
    if cpus_per_actor <= 0:
        cluster_cpus = _ray_get_cluster_cpus() or 1
        cpus_per_actor = max(
            1,
            min(
                int(_get_min_node_cpus() or 1),
                int(cluster_cpus // ray_params.num_actors)))
    return cpus_per_actor, gpus_per_actor


def _create_actor(
        rank: int,
        num_actors: int,
        num_cpus_per_actor: int,
        num_gpus_per_actor: int,
        resources_per_actor: Optional[Dict] = None,
        placement_group: Optional[PlacementGroup] = None,
        queue: Optional[Queue] = None,
        checkpoint_frequency: int = 5,
        distributed_callbacks: Optional[Sequence[DistributedCallback]] = None
) -> ActorHandle:
    # Send DEFAULT_PG here, which changed in Ray >= 1.5.0
    # If we send `None`, this will ignore the parent placement group and
    # lead to errors e.g. when used within Ray Tune
    return _RemoteRayXGBoostActor.options(
        num_cpus=num_cpus_per_actor,
        num_gpus=num_gpus_per_actor,
        resources=resources_per_actor,
        placement_group=placement_group or DEFAULT_PG).remote(
            rank=rank,
            num_actors=num_actors,
            queue=queue,
            checkpoint_frequency=checkpoint_frequency,
            distributed_callbacks=distributed_callbacks)


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


def _shutdown(actors: List[ActorHandle],
              pending_actors: Optional[Dict[int, Tuple[
                  ActorHandle, _PrepareActorTask]]] = None,
              queue: Optional[Queue] = None,
              event: Optional[Event] = None,
              placement_group: Optional[PlacementGroup] = None,
              force: bool = False):
    alive_actors = [a for a in actors if a is not None]
    if pending_actors:
        alive_actors += [a for (a, _) in pending_actors.values()]

    if force:
        for actor in alive_actors:
            ray.kill(actor)
    else:
        done_refs = [a.__ray_terminate__.remote() for a in alive_actors]
        # Wait 5 seconds for actors to die gracefully.
        done, not_done = ray.wait(done_refs, timeout=5)
        if not_done:
            # If all actors are not able to die gracefully, then kill them.
            for actor in alive_actors:
                ray.kill(actor)
    for i in range(len(actors)):
        actors[i] = None
    if queue:
        queue.shutdown()
    if event:
        event.shutdown()
    if placement_group:
        remove_placement_group(placement_group)


def _create_placement_group(cpus_per_actor, gpus_per_actor,
                            resources_per_actor, num_actors, strategy):
    resources_per_bundle = {"CPU": cpus_per_actor, "GPU": gpus_per_actor}
    extra_resources_per_bundle = {} if resources_per_actor is None else \
        resources_per_actor
    # Create placement group for training worker colocation.
    bundles = [{
        **resources_per_bundle,
        **extra_resources_per_bundle
    } for _ in range(num_actors)]
    pg = placement_group(bundles, strategy=strategy)
    # Wait for placement group to get created.
    logger.debug("Waiting for placement group to start.")
    ready, _ = ray.wait([pg.ready()], timeout=PLACEMENT_GROUP_TIMEOUT_S)
    if ready is not None:
        logger.debug("Placement group has started.")
    else:
        raise TimeoutError("Placement group creation timed out. Make sure "
                           "your cluster either has enough resources or use "
                           "an autoscaling cluster. Current resources "
                           "available: {}, resources requested by the "
                           "placement group: {}".format(
                               ray.available_resources(), pg.bundle_specs))
    return pg


def _create_communication_processes(added_tune_callback: bool = False):
    # Create Queue and Event actors and make sure to colocate with driver node.
    node_ip = get_node_ip_address()
    # Have to explicitly set num_cpus to 0.
    placement_option = {"num_cpus": 0}
    if added_tune_callback and TUNE_USING_PG:
        # If Tune is using placement groups, then we force Queue and
        # StopEvent onto same bundle as the Trainable.
        # This forces all 3 to be on the same node.
        current_pg = get_current_placement_group()
        if current_pg is None:
            # This means the user is not using Tune PGs after all -
            # e.g. via setting an environment variable.
            placement_option.update({"resources": {f"node:{node_ip}": 0.01}})
        else:
            placement_option.update({
                "placement_group": current_pg,
                "placement_group_bundle_index": 0
            })
    else:
        placement_option.update({"resources": {f"node:{node_ip}": 0.01}})
    queue = Queue(actor_options=placement_option)  # Queue actor
    stop_event = Event(actor_options=placement_option)  # Stop event actor
    return queue, stop_event


@dataclass
class _TrainingState:
    actors: List[Optional[ActorHandle]]
    queue: Queue
    stop_event: Event

    checkpoint: _Checkpoint
    additional_results: Dict

    training_started_at: float = 0.

    placement_group: Optional[PlacementGroup] = None

    failed_actor_ranks: set = field(default_factory=set)

    # Last time we checked resources to schedule new actors
    last_resource_check_at: float = 0
    pending_actors: Dict[int, Tuple[ActorHandle, _PrepareActorTask]] = field(
        default_factory=dict)
    restart_training_at: Optional[float] = None


def _train(params: Dict,
           dtrain: RayDMatrix,
           *args,
           evals=(),
           ray_params: RayParams,
           cpus_per_actor: int,
           gpus_per_actor: int,
           _training_state: _TrainingState,
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
    from xgboost_ray.elastic import _maybe_schedule_new_actors, \
        _update_scheduled_actor_states, _get_actor_alive_status

    # Un-schedule possible scheduled restarts
    _training_state.restart_training_at = None

    if "nthread" in params or "n_jobs" in params:
        if ("nthread" in params and params["nthread"] > cpus_per_actor) or (
                "n_jobs" in params and params["n_jobs"] > cpus_per_actor):
            raise ValueError(
                "Specified number of threads greater than number of CPUs. "
                "\nFIX THIS by passing a lower value for the `nthread` "
                "parameter or a higher number for `cpus_per_actor`.")
    else:
        params["nthread"] = cpus_per_actor
        params["n_jobs"] = cpus_per_actor

    # This is a callback that handles actor failures.
    # We identify the rank of the failed actor, add this to a set of
    # failed actors (which we might want to restart later), and set its
    # entry in the actor list to None.
    def handle_actor_failure(actor_id):
        rank = _training_state.actors.index(actor_id)
        _training_state.failed_actor_ranks.add(rank)
        _training_state.actors[rank] = None

    # Here we create new actors. In the first invocation of _train(), this
    # will be all actors. In future invocations, this may be less than
    # the num_actors setting, depending on the failure mode.
    newly_created = 0
    for i in list(_training_state.failed_actor_ranks):
        if _training_state.actors[i] is not None:
            raise RuntimeError(
                f"Trying to create actor with rank {i}, but it already "
                f"exists.")
        actor = _create_actor(
            rank=i,
            num_actors=ray_params.num_actors,
            num_cpus_per_actor=cpus_per_actor,
            num_gpus_per_actor=gpus_per_actor,
            resources_per_actor=ray_params.resources_per_actor,
            placement_group=_training_state.placement_group,
            queue=_training_state.queue,
            checkpoint_frequency=ray_params.checkpoint_frequency,
            distributed_callbacks=ray_params.distributed_callbacks)
        # Set actor entry in our list
        _training_state.actors[i] = actor
        # Remove from this set so it is not created again
        _training_state.failed_actor_ranks.remove(i)
        newly_created += 1

    alive_actors = sum(1 for a in _training_state.actors if a is not None)
    logger.info(f"[RayXGBoost] Created {newly_created} new actors "
                f"({alive_actors} total actors). Waiting until actors "
                f"are ready for training.")

    # For distributed datasets (e.g. Modin), this will initialize
    # (and fix) the assignment of data shards to actor ranks
    dtrain.assert_enough_shards_for_actors(num_actors=ray_params.num_actors)
    dtrain.assign_shards_to_actors(_training_state.actors)
    for deval, _ in evals:
        deval.assert_enough_shards_for_actors(num_actors=ray_params.num_actors)
        deval.assign_shards_to_actors(_training_state.actors)

    load_data = [dtrain] + [eval[0] for eval in evals]

    prepare_actor_tasks = [
        _PrepareActorTask(
            actor,
            # Maybe we got a new Queue actor, so send it to all actors.
            queue=_training_state.queue,
            # Maybe we got a new Event actor, so send it to all actors.
            stop_event=_training_state.stop_event,
            # Trigger data loading
            load_data=load_data) for actor in _training_state.actors
        if actor is not None
    ]

    start_wait = time.time()
    last_status = start_wait
    try:
        # Construct list before calling any() to force evaluation
        ready_states = [task.is_ready() for task in prepare_actor_tasks]
        while not all(ready_states):
            if time.time() >= last_status + STATUS_FREQUENCY_S:
                wait_time = time.time() - start_wait
                logger.info(f"Waiting until actors are ready "
                            f"({wait_time:.0f} seconds passed).")
                last_status = time.time()
            time.sleep(0.1)
            ready_states = [task.is_ready() for task in prepare_actor_tasks]

    except Exception as exc:
        _training_state.stop_event.set()
        _get_actor_alive_status(_training_state.actors, handle_actor_failure)
        raise RayActorError from exc

    logger.info("[RayXGBoost] Starting XGBoost training.")

    # Start Rabit tracker for gradient sharing
    rabit_process, env = _start_rabit_tracker(alive_actors)
    rabit_args = [("%s=%s" % item).encode() for item in env.items()]

    # Load checkpoint if we have one. In that case we need to adjust the
    # number of training rounds.
    if _training_state.checkpoint.value:
        kwargs["xgb_model"] = pickle.loads(_training_state.checkpoint.value)
        if _training_state.checkpoint.iteration == -1:
            # -1 means training already finished.
            logger.error(
                "Trying to load continue from checkpoint, but the checkpoint"
                "indicates training already finished. Returning last"
                "checkpointed model instead.")
            return kwargs["xgb_model"], {}, _training_state.additional_results

    # The callback_returns dict contains actor-rank indexed lists of
    # results obtained through the `put_queue` function, usually
    # sent via callbacks.
    callback_returns = _training_state.additional_results.get(
        "callback_returns")
    if callback_returns is None:
        callback_returns = [list() for _ in range(len(_training_state.actors))]
        _training_state.additional_results[
            "callback_returns"] = callback_returns

    _training_state.training_started_at = time.time()

    # Trigger the train function
    live_actors = [
        actor for actor in _training_state.actors if actor is not None
    ]
    training_futures = [
        actor.train.remote(
            rabit_args,
            i == 0,  # return_bst
            params,
            dtrain,
            evals,
            *args,
            **kwargs) for i, actor in enumerate(live_actors)
    ]

    # Failure handling loop. Here we wait until all training tasks finished.
    # If a training task fails, we stop training on the remaining actors,
    # check which ones are still alive, and raise the error.
    # The train() wrapper function will then handle the error.
    start_wait = time.time()
    last_status = start_wait
    try:
        not_ready = training_futures
        while not_ready:
            if _training_state.queue:
                _handle_queue(
                    queue=_training_state.queue,
                    checkpoint=_training_state.checkpoint,
                    callback_returns=callback_returns)

            if ray_params.elastic_training \
                    and not ELASTIC_RESTART_DISABLED:
                _maybe_schedule_new_actors(
                    training_state=_training_state,
                    num_cpus_per_actor=cpus_per_actor,
                    num_gpus_per_actor=gpus_per_actor,
                    resources_per_actor=ray_params.resources_per_actor,
                    ray_params=ray_params,
                    load_data=load_data)

                # This may raise RayXGBoostActorAvailable
                _update_scheduled_actor_states(_training_state)

            if time.time() >= last_status + STATUS_FREQUENCY_S:
                wait_time = time.time() - start_wait
                logger.info(f"Training in progress "
                            f"({wait_time:.0f} seconds since last restart).")
                last_status = time.time()

            ready, not_ready = ray.wait(
                not_ready, num_returns=len(not_ready), timeout=1)
            ray.get(ready)

        # Get items from queue one last time
        if _training_state.queue:
            _handle_queue(
                queue=_training_state.queue,
                checkpoint=_training_state.checkpoint,
                callback_returns=callback_returns)

    # The inner loop should catch all exceptions
    except Exception as exc:
        logger.debug(f"Caught exception in training loop: {exc}")

        # Stop all other actors from training
        _training_state.stop_event.set()

        # Check which actors are still alive
        _get_actor_alive_status(_training_state.actors, handle_actor_failure)

        # Todo: Try to fetch newer checkpoint, store in `_checkpoint`
        # Shut down rabit
        _stop_rabit_tracker(rabit_process)

        raise RayActorError from exc

    # Training is now complete.
    # Stop Rabit tracking process
    _stop_rabit_tracker(rabit_process)

    # Get all results from all actors.
    all_results: List[Dict[str, Any]] = ray.get(training_futures)

    # All results should be the same because of Rabit tracking. But only
    # the first one actually returns its bst object.
    bst = all_results[0]["bst"]
    evals_result = all_results[0]["evals_result"]

    if callback_returns:
        _training_state.additional_results[
            "callback_returns"] = callback_returns

    total_n = sum(res["train_n"] or 0 for res in all_results)

    _training_state.additional_results["total_n"] = total_n

    return bst, evals_result, _training_state.additional_results


def train(
        params: Dict,
        dtrain: RayDMatrix,
        num_boost_round: int = 10,
        *args,
        evals: Union[List[Tuple[RayDMatrix, str]], Tuple[RayDMatrix, str]] = (
        ),
        evals_result: Optional[Dict] = None,
        additional_results: Optional[Dict] = None,
        ray_params: Union[None, RayParams, Dict] = None,
        _remote: Optional[bool] = None,
        **kwargs) -> xgb.Booster:
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
    A maximum of ``ray_params.max_actor_restarts`` restarts will be tried
    before exiting.

    Second, if ``ray_params.elastic_training`` is ``False`` and
    the number of restarts is below ``ray_params.max_actor_restarts``,
    Ray will try to schedule the dead actor again, load the data shard
    on this actor, and then continue training from the latest checkpoint.

    Third, if none of the above is the case, training is aborted.

    Args:
        params (Dict): parameter dict passed to ``xgboost.train()``
        dtrain (RayDMatrix): Data object containing the training data.
        evals (Union[List[Tuple[RayDMatrix, str]], Tuple[RayDMatrix, str]]):
            ``evals`` tuple passed to ``xgboost.train()``.
        evals_result (Optional[Dict]): Dict to store evaluation results in.
        additional_results (Optional[Dict]): Dict to store additional results.
        ray_params (Union[None, RayParams, Dict]): Parameters to configure
            Ray-specific behavior. See :class:`RayParams` for a list of valid
            configuration parameters.
        _remote (bool): Whether to run the driver process in a remote
            function. This is enabled by default in Ray client mode.
        **kwargs: Keyword arguments will be passed to the local
            `xgb.train()` calls.

    Returns: An ``xgboost.Booster`` object.
    """
    os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

    if _remote is None:
        _remote = _is_client_connected() and \
                  not is_session_enabled()

    if not ray.is_initialized():
        ray.init()

    if _remote:
        # Run this function as a remote function to support Ray client mode.
        @ray.remote(num_cpus=0)
        def _wrapped(*args, **kwargs):
            _evals_result = {}
            _additional_results = {}
            bst = train(
                *args,
                num_boost_round=num_boost_round,
                evals_result=_evals_result,
                additional_results=_additional_results,
                **kwargs)
            return bst, _evals_result, _additional_results

        # Make sure that train is called on the server node.
        _wrapped = force_on_current_node(_wrapped)

        bst, train_evals_result, train_additional_results = ray.get(
            _wrapped.remote(
                params,
                dtrain,
                *args,
                evals=evals,
                ray_params=ray_params,
                _remote=False,
                **kwargs,
            ))
        if isinstance(evals_result, dict):
            evals_result.update(train_evals_result)
        if isinstance(additional_results, dict):
            additional_results.update(train_additional_results)
        return bst

    _maybe_print_legacy_warning()

    start_time = time.time()

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

    added_tune_callback = _try_add_tune_callback(kwargs)
    # Tune currently does not support elastic training.
    if added_tune_callback and ray_params.elastic_training and not bool(
            os.getenv("RXGB_ALLOW_ELASTIC_TUNE", "0")):
        raise ValueError("Elastic Training cannot be used with Ray Tune. "
                         "Please disable elastic_training in RayParams in "
                         "order to use xgboost_ray with Tune.")

    if added_tune_callback:
        # Don't autodetect resources when used with Tune.
        cpus_per_actor = ray_params.cpus_per_actor
        gpus_per_actor = max(0, ray_params.gpus_per_actor)
    else:
        cpus_per_actor, gpus_per_actor = _autodetect_resources(
            ray_params=ray_params,
            use_tree_method="tree_method" in params
            and params["tree_method"] is not None
            and params["tree_method"].startswith("gpu"))

    tree_method = params.get("tree_method", "auto") or "auto"

    # preemptively raise exceptions with bad params
    if tree_method == "exact":
        raise ValueError(
            "`exact` tree method doesn't support distributed training.")

    if params.get("updater", None) == "grow_colmaker":
        raise ValueError(
            "`grow_colmaker` updater doesn't support distributed training.")

    if gpus_per_actor > 0 and not tree_method.startswith("gpu_"):
        warnings.warn(
            f"GPUs have been assigned to the actors, but the current XGBoost "
            f"tree method is set to `{tree_method}`. Thus, GPUs will "
            f"currently not be used. To enable GPUs usage, please set the "
            f"`tree_method` to a GPU-compatible option, "
            f"e.g. `gpu_hist`.")

    if gpus_per_actor == 0 and cpus_per_actor == 0:
        raise ValueError("cpus_per_actor and gpus_per_actor both cannot be "
                         "0. Are you sure your cluster has CPUs available?")

    if ray_params.elastic_training and ray_params.max_failed_actors == 0:
        raise ValueError(
            "Elastic training enabled but the maximum number of failed "
            "actors is set to 0. This means that elastic training is "
            "effectively disabled. Please set `RayParams.max_failed_actors` "
            "to something larger than 0 to enable elastic training.")

    if ray_params.elastic_training and ray_params.max_actor_restarts == 0:
        raise ValueError(
            "Elastic training enabled but the maximum number of actor "
            "restarts is set to 0. This means that elastic training is "
            "effectively disabled. Please set `RayParams.max_actor_restarts` "
            "to something larger than 0 to enable elastic training.")

    if not dtrain.has_label:
        raise ValueError(
            "Training data has no label set. Please make sure to set "
            "the `label` argument when initializing `RayDMatrix()` "
            "for data you would like to train on.")

    if not dtrain.loaded and not dtrain.distributed:
        dtrain.load_data(ray_params.num_actors)

    for (deval, name) in evals:
        if not deval.has_label:
            raise ValueError(
                "Evaluation data has no label set. Please make sure to set "
                "the `label` argument when initializing `RayDMatrix()` "
                "for data you would like to evaluate on.")
        if not deval.loaded and not deval.distributed:
            deval.load_data(ray_params.num_actors)

    bst = None
    train_evals_result = {}
    train_additional_results = {}

    tries = 0
    checkpoint = _Checkpoint()  # Keep track of latest checkpoint
    current_results = {}  # Keep track of additional results
    actors = [None] * ray_params.num_actors  # All active actors
    pending_actors = {}

    # Create the Queue and Event actors.
    queue, stop_event = _create_communication_processes(added_tune_callback)

    placement_strategy = None
    if not ray_params.elastic_training:
        if added_tune_callback:
            if TUNE_USING_PG:
                # If Tune is using placement groups, then strategy has already
                # been set. Don't create an additional placement_group here.
                placement_strategy = None
            else:
                placement_strategy = "PACK"
        elif bool(_USE_SPREAD_STRATEGY):
            placement_strategy = "SPREAD"

    if placement_strategy is not None:
        pg = _create_placement_group(cpus_per_actor, gpus_per_actor,
                                     ray_params.resources_per_actor,
                                     ray_params.num_actors, placement_strategy)
    else:
        pg = None

    start_actor_ranks = set(range(ray_params.num_actors))  # Start these

    total_training_time = 0.
    boost_rounds_left = num_boost_round
    last_checkpoint_value = checkpoint.value
    while tries <= max_actor_restarts:
        # Only update number of iterations if the checkpoint changed
        # If it didn't change, we already subtracted the iterations.
        if checkpoint.iteration >= 0 and \
                checkpoint.value != last_checkpoint_value:
            boost_rounds_left -= checkpoint.iteration + 1

        last_checkpoint_value = checkpoint.value

        logger.debug(f"Boost rounds left: {boost_rounds_left}")

        training_state = _TrainingState(
            actors=actors,
            queue=queue,
            stop_event=stop_event,
            checkpoint=checkpoint,
            additional_results=current_results,
            training_started_at=0.,
            placement_group=pg,
            failed_actor_ranks=start_actor_ranks,
            pending_actors=pending_actors)

        try:
            bst, train_evals_result, train_additional_results = _train(
                params,
                dtrain,
                boost_rounds_left,
                *args,
                evals=evals,
                ray_params=ray_params,
                cpus_per_actor=cpus_per_actor,
                gpus_per_actor=gpus_per_actor,
                _training_state=training_state,
                **kwargs)
            if training_state.training_started_at > 0.:
                total_training_time += time.time(
                ) - training_state.training_started_at
            break
        except (RayActorError, RayTaskError) as exc:
            if training_state.training_started_at > 0.:
                total_training_time += time.time(
                ) - training_state.training_started_at
            alive_actors = sum(1 for a in actors if a is not None)
            start_again = False
            if ray_params.elastic_training:
                if alive_actors < ray_params.num_actors - \
                        ray_params.max_failed_actors:
                    raise RuntimeError(
                        "A Ray actor died during training and the maximum "
                        "number of dead actors in elastic training was "
                        "reached. Shutting down training.") from exc

                # Do not start new actors before resuming training
                # (this might still restart actors during training)
                start_actor_ranks.clear()

                if exc.__cause__ and isinstance(exc.__cause__,
                                                RayXGBoostActorAvailable):
                    # New actor available, integrate into training loop
                    logger.info(
                        f"A new actor became available. Re-starting training "
                        f"from latest checkpoint with new actor. "
                        f"This will use {alive_actors} existing actors and "
                        f"start {len(start_actor_ranks)} new actors. "
                        f"Sleeping for 10 seconds for cleanup.")
                    tries -= 1  # This is deliberate so shouldn't count
                    start_again = True

                elif tries + 1 <= max_actor_restarts:
                    if exc.__cause__ and isinstance(exc.__cause__,
                                                    RayXGBoostTrainingError):
                        logger.warning(f"Caught exception: {exc.__cause__}")
                    logger.warning(
                        f"A Ray actor died during training. Trying to "
                        f"continue training on the remaining actors. "
                        f"This will use {alive_actors} existing actors and "
                        f"start {len(start_actor_ranks)} new actors. "
                        f"Sleeping for 10 seconds for cleanup.")
                    start_again = True

            elif tries + 1 <= max_actor_restarts:
                if exc.__cause__ and isinstance(exc.__cause__,
                                                RayXGBoostTrainingError):
                    logger.warning(f"Caught exception: {exc.__cause__}")
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
                time.sleep(5)
                queue, stop_event = _create_communication_processes()
            else:
                raise RuntimeError(
                    f"A Ray actor died during training and the maximum number "
                    f"of retries ({max_actor_restarts}) is exhausted."
                ) from exc
            tries += 1

    total_time = time.time() - start_time

    train_additional_results["training_time_s"] = total_training_time
    train_additional_results["total_time_s"] = total_time

    logger.info("[RayXGBoost] Finished XGBoost training on training data "
                "with total N={total_n:,} in {total_time_s:.2f} seconds "
                "({training_time_s:.2f} pure XGBoost training time).".format(
                    **train_additional_results))

    _shutdown(
        actors=actors,
        pending_actors=pending_actors,
        queue=queue,
        event=stop_event,
        placement_group=pg,
        force=False)

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
            resources_per_actor=ray_params.resources_per_actor,
            distributed_callbacks=ray_params.distributed_callbacks)
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
            _remote: Optional[bool] = None,
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
        _remote (bool): Whether to run the driver process in a remote
            function. This is enabled by default in Ray client mode.
        **kwargs: Keyword arguments will be passed to the local
            `xgb.predict()` calls.

    Returns: ``np.ndarray`` containing the predicted labels.

    """
    os.environ.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")

    if _remote is None:
        _remote = _is_client_connected() and \
                  not is_session_enabled()

    if not ray.is_initialized():
        ray.init()

    if _remote:
        return ray.get(
            ray.remote(num_cpus=0)(predict).remote(
                model, data, ray_params, _remote=False, **kwargs))

    _maybe_print_legacy_warning()

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
