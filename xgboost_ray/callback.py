from abc import ABC
from typing import Dict, Sequence, TYPE_CHECKING, Any, Union

import os
import pandas as pd

if TYPE_CHECKING:
    from xgboost_ray.main import RayXGBoostActor
    from xgboost_ray.matrix import RayDMatrix


class DistributedCallback(ABC):
    """Distributed callbacks for RayXGBoostActors.

    The hooks of these callbacks are executed on the remote Ray actors
    at different points in time. They can be used to set environment
    variables or to prepare the training/prediction environment in other
    ways. Distributed callback objects are de-serialized on each actor
    and are then independent of each other - changing the state of one
    callback will not alter the state of the other copies on different actors.

    Callbacks can be passed to xgboost_ray via
    :class:`RayParams <xgboost_ray.main.RayParams>` using the
    ``distributed_callbacks`` parameter.
    """

    def on_init(self, actor: "RayXGBoostActor", *args, **kwargs):
        pass

    def before_data_loading(self, actor: "RayXGBoostActor", data: "RayDMatrix",
                            *args, **kwargs):
        pass

    def after_data_loading(self, actor: "RayXGBoostActor", data: "RayDMatrix",
                           *args, **kwargs):
        pass

    def before_train(self, actor: "RayXGBoostActor", *args, **kwargs):
        pass

    def after_train(self, actor: "RayXGBoostActor", result_dict: Dict, *args,
                    **kwargs):
        pass

    def before_predict(self, actor: "RayXGBoostActor", *args, **kwargs):
        pass

    def after_predict(self, actor: "RayXGBoostActor",
                      predictions: Union[pd.Series, pd.DataFrame], *args,
                      **kwargs):
        pass


class DistributedCallbackContainer:
    def __init__(self, callbacks: Sequence[DistributedCallback]):
        self.callbacks = callbacks or []

    def on_init(self, actor: "RayXGBoostActor", *args, **kwargs):
        for callback in self.callbacks:
            callback.on_init(actor, *args, **kwargs)

    def before_data_loading(self, actor: "RayXGBoostActor", data: "RayDMatrix",
                            *args, **kwargs):
        for callback in self.callbacks:
            callback.before_data_loading(actor, data, *args, **kwargs)

    def after_data_loading(self, actor: "RayXGBoostActor", data: "RayDMatrix",
                           *args, **kwargs):
        for callback in self.callbacks:
            callback.after_data_loading(actor, data, *args, **kwargs)

    def before_train(self, actor: "RayXGBoostActor", *args, **kwargs):
        for callback in self.callbacks:
            callback.before_train(actor, *args, **kwargs)

    def after_train(self, actor: "RayXGBoostActor", result_dict: Dict, *args,
                    **kwargs):
        for callback in self.callbacks:
            callback.after_train(actor, result_dict, *args, **kwargs)

    def before_predict(self, actor: "RayXGBoostActor", *args, **kwargs):
        for callback in self.callbacks:
            callback.before_predict(actor, *args, **kwargs)

    def after_predict(self, actor: "RayXGBoostActor",
                      predictions: Union[pd.Series, pd.DataFrame], *args,
                      **kwargs):
        for callback in self.callbacks:
            callback.after_predict(actor, predictions, *args, **kwargs)


class EnvironmentCallback(DistributedCallback):
    def __init__(self, env_dict: Dict[str, Any]):
        self.env_dict = env_dict

    def on_init(self, actor, *args, **kwargs):
        os.environ.update(self.env_dict)
