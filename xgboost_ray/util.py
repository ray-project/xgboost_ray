from typing import Dict

import ray
import asyncio

from ray.util.queue import Queue as RayQueue, _QueueActor


@ray.remote
class _EventActor:
    def __init__(self):
        self._event = asyncio.Event()

    def set(self):
        self._event.set()

    def clear(self):
        self._event.clear()

    def is_set(self):
        return self._event.is_set()


class Event:
    def __init__(self, actor_options: Dict = {}):
        self.actor = _EventActor.options(**actor_options).remote()

    def set(self):
        self.actor.set.remote()

    def clear(self):
        self.actor.clear.remote()

    def is_set(self):
        return ray.get(self.actor.is_set.remote())

    def shutdown(self):
        if self.actor:
            ray.kill(self.actor)
        self.actor = None

# TODO: Propogate this into RayQueue.
class Queue(RayQueue):
    def __init__(self, maxsize: int = 0, actor_options: Dict = {}) -> None:
        self.maxsize = maxsize
        self.actor = _QueueActor.options(**actor_options).remote(self.maxsize)

    def shutdown(self):
        if self.actor:
            ray.kill(self.actor)
        self.actor = None
