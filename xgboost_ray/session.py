from typing import Optional
from ray.util.queue import Queue


class RayXGBoostSession:
    def __init__(self, rank: int, queue: Optional[Queue]):
        self._rank = rank
        self._queue = queue

    def get_actor_rank(self):
        return self._rank

    def set_queue(self, queue):
        self._queue = queue

    def put_queue(self, item):
        if self._queue is None:
            raise ValueError(
                "Trying to put something into session queue, but queue "
                "was not initialized. This is probably a bug, please raise "
                "an issue at https://github.com/ray-project/xgboost_ray")
        self._queue.put((self._rank, item))


_session = None


def init_session(*args, **kwargs):
    global _session
    if _session:
        raise ValueError(
            "Trying to initialize RayXGBoostSession twice."
            "\nFIX THIS by not calling `init_session()` manually.")
    _session = RayXGBoostSession(*args, **kwargs)


def get_session() -> RayXGBoostSession:
    global _session
    if not _session or not isinstance(_session, RayXGBoostSession):
        raise ValueError(
            "Trying to access RayXGBoostSession from outside an XGBoost run."
            "\nFIX THIS by calling function in `session.py` like "
            "`get_actor_rank()` only from within an XGBoost actor session.")
    return _session


def set_session_queue(queue: Queue):
    session = get_session()
    session.set_queue(queue)


def get_actor_rank() -> int:
    session = get_session()
    return session.get_actor_rank()


def get_rabit_rank() -> int:
    import xgboost as xgb
    return xgb.rabit.get_rank()


def put_queue(*args, **kwargs):
    session = get_session()
    session.put_queue(*args, **kwargs)
