from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xgboost
    from xgboost.core import XGBoostError, EarlyStopException
else:
    try:
        import xgboost
        from xgboost.core import XGBoostError, EarlyStopException
    except ImportError:
        xgboost = XGBoostError = EarlyStopException = None

__all__ = ["xgboost", "XGBoostError", "EarlyStopException"]
