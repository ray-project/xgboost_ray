import os
import time
from collections import defaultdict
from typing import Dict, Tuple, Set

import ray
from ray.actor import ActorHandle

from xgboost_ray.callback import DistributedCallback
from xgboost_ray.compat import TrainingCallback
from xgboost_ray.session import get_actor_rank


@ray.remote(num_cpus=0)
class FaultToleranceManager:
    def __init__(self, start_boost_round: int = 0):
        self.global_boost_round = start_boost_round

        # Dict from boost_round -> actor ranks to die
        self.scheduled_kill: Dict[int, Set[int]] = defaultdict(set)

        # Dict from actor rank -> starts/ends of boost rounds to sleep
        self.delayed_return: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

        # List of tuples (global_boost_round, actor_boost_round) to log
        # actor iterations
        self.training_logs = defaultdict(list)

    def schedule_kill(self, rank: int, boost_round: int):
        """Kill an actor when reaching this global boost round"""
        self.scheduled_kill[boost_round].add(rank)

    def delay_return(self, rank: int, start_boost_round: int,
                     end_boost_round: int):
        """Do not allow an actor to finish data loading between these rounds"""
        self.delayed_return[rank].add((start_boost_round, end_boost_round))

    def inc_boost_round(self, rank: int):
        """Increase global boosting round"""
        if rank == 0:
            self.global_boost_round += 1

    def log_iteration(self, rank: int, boost_round: int):
        """Log iteration"""
        self.training_logs[rank].append((self.global_boost_round, boost_round))

    def should_die(self, rank: int):
        """Returns True if the actor should terminate the training job now."""
        die = False
        for round in range(self.global_boost_round + 1):
            # Loop through all rounds until now to deal with race conditions
            if rank in self.scheduled_kill[round]:
                self.scheduled_kill[round].remove(rank)
                die = True
        return die

    def should_sleep(self, rank: int):
        """Returns True if the actor should not finish data loading, yet."""
        if self.delayed_return[rank]:
            for start, end in self.delayed_return[rank]:
                if start <= self.global_boost_round < end:
                    return True
        return False

    def get_logs(self):
        return self.training_logs


class DelayedLoadingCallback(DistributedCallback):
    """Used to control when actors return to training"""

    def __init__(self,
                 ft_manager: ActorHandle,
                 reload_data=True,
                 sleep_time=0.5):
        self.ft_manager = ft_manager
        self.reload_data = reload_data
        self.sleep_time = sleep_time

    def after_data_loading(self, actor, data, *args, **kwargs):
        print(f"Rank {actor.rank} - after load")
        while ray.get(self.ft_manager.should_sleep.remote(actor.rank)):
            time.sleep(self.sleep_time)
        print(f"Rank {actor.rank} - returning now")


class DieCallback(TrainingCallback):
    """Used to control when actors should die during training.

    Also can add delay to each boosting round.
    """

    def __init__(self, ft_manager: ActorHandle, training_delay: float = 0):
        self.ft_manager = ft_manager
        self.training_delay = training_delay
        super(DieCallback, self).__init__()

    def before_iteration(self, model, epoch, evals_log):
        if ray.get(self.ft_manager.should_die.remote(get_actor_rank())):
            pid = os.getpid()
            print(f"Killing process: {pid}")
            print(f"Rank {get_actor_rank()} will now die.")
            time.sleep(1)
            os.kill(pid, 9)
            time.sleep(10)  # Don't continue training, just die

    def after_iteration(self, model, epoch, evals_log):
        # ray.get to make sure this is up to date in the next iteration
        ray.get(self.ft_manager.log_iteration.remote(get_actor_rank(), epoch))
        if self.training_delay > 0:
            time.sleep(self.training_delay)
        if get_actor_rank() == 0:
            ray.get(self.ft_manager.inc_boost_round.remote(get_actor_rank()))
