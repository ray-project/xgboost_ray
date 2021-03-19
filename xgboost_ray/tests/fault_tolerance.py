from collections import defaultdict
from typing import List, Dict, Tuple

from ray.actor import ActorHandle

from xgboost_ray.callback import DistributedCallback


class FaultToleranceManager:
    def __init__(self, start_boost_round: int = 0):
        self.global_boost_round = start_boost_round

        # Dict from boost_round -> actor ranks to die
        self.scheduled_kill: Dict[int, List[int]] = defaultdict(list)

        # Dict from actor rank -> starts/ends of boost rounds to sleep
        self.delayed_return: Dict[int, List[Tuple[int, int]]] = defaultdict(
            list)

    def schedule_kill(self, rank: int, boost_round: int):
        """Kill an actor when reaching this global boost round"""
        pass

    def delay_return(self, rank: int, start_boost_round: int,
                     end_boost_round: int):
        """Do not allow an actor to finish data loading between these rounds"""
        pass

    def inc_boost_round(self, rank: int):
        """Increase global boost round. Only allow actor 0 to do that."""
        if rank == 0:
            self.global_boost_round += 1

    def should_die(self, rank: int, boost_round: int):
        """Returns True if the actor should terminate the training job now."""
        if rank in self.scheduled_kill[boost_round]:
            self.scheduled_kill[boost_round].remove(rank)
            return True
        return False

    def should_sleep(self, rank: int):
        """Returns True if the actor should not finish data loading, yet."""
        if self.delayed_return[rank]:
            for start, end in self.delayed_return[rank]:
                if start <= self.global_boost_round < end:
                    return True
        return False


class DelayedLoadingCallback(DistributedCallback):
    def __init__(self, ft_manager: ActorHandle):
        self.ft_manager = ft_manager

    def after_data_loading(self, actor, data, *args, **kwargs):
        import time

        while self.ft_manager.should_sleep.remote(actor.rank):
            time.sleep(1)
