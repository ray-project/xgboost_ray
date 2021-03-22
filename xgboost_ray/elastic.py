import time
from typing import Optional, Dict, List, Tuple, Callable

import ray

from xgboost_ray.main import RayParams, _TrainingState, \
    logger, ActorHandle, _PrepareActorTask, _create_actor, \
    RayXGBoostActorAvailable, \
    ELASTIC_RESTART_RESOURCE_CHECK_S, ELASTIC_RESTART_GRACE_PERIOD_S

from xgboost_ray.matrix import RayDMatrix


def _maybe_schedule_new_actors(
        training_state: _TrainingState, num_cpus_per_actor: int,
        num_gpus_per_actor: int, resources_per_actor: Optional[Dict],
        ray_params: RayParams, load_data: List[RayDMatrix]) -> bool:
    """Schedule new actors for elastic training if resources are available.

    Potentially starts new actors and triggers data loading."""

    # This is only enabled for elastic training.
    if not ray_params.elastic_training:
        return False

    missing_actor_ranks = [
        rank for rank, actor in enumerate(training_state.actors)
        if actor is None and rank not in training_state.pending_actors
    ]

    # If all actors are alive, there is nothing to do.
    if not missing_actor_ranks:
        return False

    now = time.time()

    # Check periodically every n seconds.
    if now < training_state.last_resource_check_at + \
            ELASTIC_RESTART_RESOURCE_CHECK_S:
        return False

    training_state.last_resource_check_at = now

    new_pending_actors: Dict[int, Tuple[ActorHandle, _PrepareActorTask]] = {}
    for rank in missing_actor_ranks:
        # Actor rank should not be already pending
        if rank in training_state.pending_actors \
                or rank in new_pending_actors:
            continue

        # Try to schedule this actor
        actor = _create_actor(
            rank=rank,
            num_actors=ray_params.num_actors,
            num_cpus_per_actor=num_cpus_per_actor,
            num_gpus_per_actor=num_gpus_per_actor,
            resources_per_actor=resources_per_actor,
            placement_group=training_state.placement_group,
            queue=training_state.queue,
            checkpoint_frequency=ray_params.checkpoint_frequency,
            distributed_callbacks=ray_params.distributed_callbacks)

        task = _PrepareActorTask(
            actor,
            queue=training_state.queue,
            stop_event=training_state.stop_event,
            load_data=load_data)

        new_pending_actors[rank] = (actor, task)
        logger.debug(f"Re-scheduled actor with rank {rank}. Waiting for "
                     f"placement and data loading before promoting it "
                     f"to training.")
    if new_pending_actors:
        training_state.pending_actors.update(new_pending_actors)
        logger.info(f"Re-scheduled {len(new_pending_actors)} actors for "
                    f"training. Once data loading finished, they will be "
                    f"integrated into training again.")
    return bool(new_pending_actors)


def _update_scheduled_actor_states(training_state: _TrainingState):
    """Update status of scheduled actors in elastic training.

    If actors finished their preparation tasks, promote them to
    proper training actors (set the `training_state.actors` entry).

    Also schedule a `RayXGBoostActorAvailable` exception so that training
    is restarted with the new actors.

    """
    now = time.time()
    actor_became_ready = False

    # Wrap in list so we can alter the `training_state.pending_actors` dict
    for rank in list(training_state.pending_actors.keys()):
        actor, task = training_state.pending_actors[rank]
        if task.is_ready():
            # Promote to proper actor
            training_state.actors[rank] = actor
            del training_state.pending_actors[rank]
            actor_became_ready = True

    if actor_became_ready:
        if not training_state.pending_actors:
            # No other actors are pending, so let's restart right away.
            training_state.restart_training_at = now - 1.

        # If an actor became ready but other actors are pending, we wait
        # for n seconds before restarting, as chances are that they become
        # ready as well (e.g. if a large node came up).
        grace_period = ELASTIC_RESTART_GRACE_PERIOD_S
        if training_state.restart_training_at is None:
            logger.debug(
                f"A RayXGBoostActor became ready for training. Waiting "
                f"{grace_period} seconds before triggering training restart.")
            training_state.restart_training_at = now + grace_period

    if training_state.restart_training_at is not None:
        if now > training_state.restart_training_at:
            training_state.restart_training_at = None
            raise RayXGBoostActorAvailable(
                "A new RayXGBoostActor became available for training. "
                "Triggering restart.")


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
                logger.debug(f"Actor {actors[rank]} with PID {pid} is alive.")
                alive += 1
            except Exception:
                rank = obj_to_rank[obj]
                logger.debug(f"Actor {actors[rank]} is _not_ alive.")
                dead += 1
                callback(actors[rank])
    logger.info(f"Actor status: {alive} alive, {dead} dead "
                f"({alive+dead} total)")

    return alive, dead
