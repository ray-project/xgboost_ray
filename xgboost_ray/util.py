from typing import Dict, Optional, List

import asyncio

import ray
from ray.util.queue import Queue as RayQueue, Empty, Full


class Unavailable:
    """No object should be instance of this class"""

    def __init__(self):
        raise RuntimeError("This class should never be instantiated.")


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
    def __init__(self, actor_options: Optional[Dict] = None):
        actor_options = {} if not actor_options else actor_options
        self.actor = ray.remote(_EventActor).options(**actor_options).remote()

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


# Remove after Ray 1.2 release.
if getattr(RayQueue, "shutdown", None) is not None:
    Queue = RayQueue
    from ray.util.queue import _QueueActor
else:
    # Have to copy the class here so that we can subclass this for mocking.
    # If we have the @ray.remote decorator, then we can't subclass it.
    class _QueueActor:
        def __init__(self, maxsize):
            self.maxsize = maxsize
            self.queue = asyncio.Queue(self.maxsize)

        def qsize(self):
            return self.queue.qsize()

        def empty(self):
            return self.queue.empty()

        def full(self):
            return self.queue.full()

        async def put(self, item, timeout=None):
            try:
                await asyncio.wait_for(self.queue.put(item), timeout)
            except asyncio.TimeoutError:
                raise Full

        async def get(self, timeout=None):
            try:
                return await asyncio.wait_for(self.queue.get(), timeout)
            except asyncio.TimeoutError:
                raise Empty

        def put_nowait(self, item):
            self.queue.put_nowait(item)

        def put_nowait_batch(self, items):
            # If maxsize is 0, queue is unbounded, so no need to check size.
            if self.maxsize > 0 and len(items) + self.qsize() > self.maxsize:
                raise Full(f"Cannot add {len(items)} items to queue of size "
                           f"{self.qsize()} and maxsize {self.maxsize}.")
            for item in items:
                self.queue.put_nowait(item)

        def get_nowait(self):
            return self.queue.get_nowait()

        def get_nowait_batch(self, num_items):
            if num_items > self.qsize():
                raise Empty(f"Cannot get {num_items} items from queue of size "
                            f"{self.qsize()}.")
            return [self.queue.get_nowait() for _ in range(num_items)]

    class Queue(RayQueue):
        def __init__(self,
                     maxsize: int = 0,
                     actor_options: Optional[Dict] = None) -> None:
            actor_options = {} if not actor_options else actor_options
            self.maxsize = maxsize
            self.actor = ray.remote(_QueueActor).options(
                **actor_options).remote(self.maxsize)

        def shutdown(self):
            if self.actor:
                ray.kill(self.actor)
            self.actor = None


class MultiActorTask:
    """Utility class to hold multiple futures.

    The `is_ready()` method will return True once all futures are ready.

    Args:
        pending_futures (list): List of object references (futures)
            that should be tracked.
    """

    def __init__(self, pending_futures: Optional[List[ray.ObjectRef]] = None):
        self._pending_futures = pending_futures or []
        self._ready_futures = []

    def is_ready(self):
        if not self._pending_futures:
            return True

        ready = True
        while ready:
            ready, not_ready = ray.wait(self._pending_futures, timeout=0)
            if ready:
                for obj in ready:
                    self._pending_futures.remove(obj)
                    self._ready_futures.append(obj)

        return not bool(self._pending_futures)
