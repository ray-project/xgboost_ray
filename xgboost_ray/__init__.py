from .main import RayParams, train, predict
from .matrix import RayDMatrix, RayDeviceQuantileDMatrix,\
    RayFileType, RayShardingMode, \
    Data, combine_data
# workaround for legacy xgboost==0.9.0
try:
    from .sklearn import RayXGBClassifier, RayXGBRegressor, \
        RayXGBRFClassifier, RayXGBRFRegressor, RayXGBRanker
except ImportError as e:
    if "WILL NOT WORK" in str(e):

        class ImportErrorOnInit:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "xgboost package is not installed. XGBoost-Ray WILL "
                    "NOT WORK. FIX THIS by running `pip install "
                    "\"xgboost-ray[default]\"`.")

        RayXGBClassifier = RayXGBRegressor = RayXGBRFClassifier = \
            RayXGBRFRegressor = RayXGBRanker = ImportErrorOnInit

__version__ = "0.1.4"

__all__ = [
    "__version__", "RayParams", "RayDMatrix", "RayDeviceQuantileDMatrix",
    "RayFileType", "RayShardingMode", "Data", "combine_data", "train",
    "predict", "RayXGBClassifier", "RayXGBRegressor", "RayXGBRFClassifier",
    "RayXGBRFRegressor", "RayXGBRanker"
]
