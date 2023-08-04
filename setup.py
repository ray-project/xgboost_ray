from setuptools import find_packages, setup

setup(
    name="xgboost_ray",
    packages=find_packages(where=".", include="xgboost_ray*"),
    version="0.1.18",
    author="Ray Team",
    description="A Ray backend for distributed XGBoost",
    license="Apache 2.0",
    long_description="A distributed backend for XGBoost built on top of "
    "distributed computing framework Ray.",
    url="https://github.com/ray-project/xgboost_ray",
    install_requires=[
        "ray>=2.0",
        "numpy>=1.16",
        "pandas",
        "wrapt>=1.12.1",
        "xgboost>=0.90",
        "packaging",
    ],
)
