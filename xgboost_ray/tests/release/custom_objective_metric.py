import ray

from xgboost_ray.tests.test_xgboost_api import XGBoostAPITest


class XGBoostDistributedAPITest(XGBoostAPITest):
    def _init_ray(self):
        self.kwargs = {"resources_per_actor": {"actor_cpus": 4.0}}
        if not ray.is_initialized():
            ray.init(address="auto")


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
