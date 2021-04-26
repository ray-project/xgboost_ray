import xgboost as xgb

try:
    from xgboost.callback import TrainingCallback
    LEGACY_CALLBACK = False
except ImportError:

    class TrainingCallback:
        def __init__(self):
            if hasattr(self, "before_iteration"):
                # XGBoost < 1.0 is looking up __dict__ to see if a
                # callback should be called before or after an iteration.
                # So here we move this to self._before_iteration and
                # overwrite the dict.
                self._before_iteration = getattr(self, "before_iteration")
                self.__dict__["before_iteration"] = True

        def __call__(self, callback_env: xgb.core.CallbackEnv):
            if hasattr(self, "_before_iteration"):
                self._before_iteration(
                    model=callback_env.model,
                    epoch=callback_env.iteration,
                    evals_log=callback_env.evaluation_result_list)

            if hasattr(self, "after_iteration"):
                self.after_iteration(
                    model=callback_env.model,
                    epoch=callback_env.iteration,
                    evals_log=callback_env.evaluation_result_list)

        def before_training(self, model):
            pass

        def after_training(self, model):
            pass

    LEGACY_CALLBACK = True

try:
    from xgboost import RabitTracker
except ImportError:
    from xgboost_ray.compat.tracker import RabitTracker

__all__ = ["TrainingCallback", "RabitTracker"]
