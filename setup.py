# PUBLISHING INSTRUCTIONS
# 1. run RXGB_SETUP_GBDT=1 python setup.py clean --all bdist_wheel
# 2. publish the gbtd_ray package
# 3. run RXGB_SETUP_GBDT=0 python setup.py clean --all bdist_wheel
# 4. publish the xgboost_ray package

import os
from setuptools import find_packages, setup

setup_kwargs = dict(
    version="0.1.3",
    author="Ray Team",
    description="A Ray backend for distributed XGBoost",
    long_description="A distributed backend for XGBoost built on top of "
    "distributed computing framework Ray.",
    url="https://github.com/ray-project/xgboost_ray",
)

# pyarrow<5.0.0 pinned until petastorm is updated
base_requirements = [
    "ray", "numpy>=1.16,<1.20", "pandas", "pyarrow<5.0.0", "wrapt>=1.12.1"
]

if bool(int(os.environ.get("RXGB_SETUP_GBDT", "1"))):
    setup(
        name="gbtd_ray",
        packages=find_packages(where=".", include="xgboost_ray*"),
        install_requires=base_requirements,
        **setup_kwargs)
else:
    setup(
        name="xgboost_ray",
        install_requires=[
            "xgboost>=0.90", f"gbtd_ray=={setup_kwargs['version']}"
        ],
        **setup_kwargs)
