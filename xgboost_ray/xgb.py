from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xgboost
else:
    try:
        import xgboost
    except ImportError:
        xgboost = None

__all__ = ["xgboost"]
