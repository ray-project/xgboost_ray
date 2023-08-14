from xgboost_ray.main import RayParams, predict, train
from xgboost_ray.matrix import (
    Data,
    RayDeviceQuantileDMatrix,
    RayDMatrix,
    RayFileType,
    RayShardingMode,
    combine_data,
)

# workaround for legacy xgboost==0.9.0
try:
    from xgboost_ray.sklearn import (
        RayXGBClassifier,
        RayXGBRanker,
        RayXGBRegressor,
        RayXGBRFClassifier,
        RayXGBRFRegressor,
    )
except ImportError:
    pass

__version__ = "0.1.18"

__all__ = [
    "__version__",
    "RayParams",
    "RayDMatrix",
    "RayDeviceQuantileDMatrix",
    "RayFileType",
    "RayShardingMode",
    "Data",
    "combine_data",
    "train",
    "predict",
    "RayXGBClassifier",
    "RayXGBRegressor",
    "RayXGBRFClassifier",
    "RayXGBRFRegressor",
    "RayXGBRanker",
]
