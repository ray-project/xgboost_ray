from xgboost_ray.main import RayParams, train, predict
from xgboost_ray.matrix import RayDMatrix, RayDeviceQuantileDMatrix,\
    RayFileType, RayShardingMode, \
    Data, combine_data
# workaround for legacy xgboost==0.9.0
try:
    from xgboost_ray.sklearn import RayXGBClassifier, RayXGBRegressor, \
        RayXGBRFClassifier, RayXGBRFRegressor, RayXGBRanker
except ImportError:
    pass

__version__ = "0.1.2"

__all__ = [
    "__version__", "RayParams", "RayDMatrix", "RayDeviceQuantileDMatrix",
    "RayFileType", "RayShardingMode", "Data", "combine_data", "train",
    "predict", "RayXGBClassifier", "RayXGBRegressor", "RayXGBRFClassifier",
    "RayXGBRFRegressor", "RayXGBRanker"
]
