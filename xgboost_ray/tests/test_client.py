import os

import pytest

import ray
from ray.util.client.ray_client_helpers import ray_start_client_server
from xgboost_ray.data_sources.ray_dataset import RAY_DATASET_AVAILABLE


@pytest.fixture
def start_client_server_4_cpus():
    ray.init(num_cpus=4)
    with ray_start_client_server() as client:
        yield client


@pytest.fixture
def start_client_server_5_cpus():
    ray.init(num_cpus=5)
    with ray_start_client_server() as client:
        yield client


def test_simple_train(start_client_server_4_cpus):
    assert ray.util.client.ray.is_connected()
    from xgboost_ray.examples.simple import main
    main(num_actors=4, cpus_per_actor=1)


@pytest.mark.skipif(
    os.environ.get("TUNE", "0") != "1", reason="Sipping Tune tests")
def test_simple_tune(start_client_server_4_cpus):
    assert ray.util.client.ray.is_connected()
    from xgboost_ray.examples.simple_tune import main
    main(cpus_per_actor=1, num_actors=1, num_samples=4)


def test_simple_dask(start_client_server_5_cpus):
    assert ray.util.client.ray.is_connected()
    from xgboost_ray.examples.simple_dask import main
    main(cpus_per_actor=1, num_actors=4)


def test_simple_modin(start_client_server_5_cpus):
    assert ray.util.client.ray.is_connected()
    from xgboost_ray.examples.simple_modin import main
    main(cpus_per_actor=1, num_actors=4)


@pytest.mark.skipif(
    not RAY_DATASET_AVAILABLE,
    reason="Ray datasets are not available in this version of Ray")
def test_simple_ray_dataset(start_client_server_5_cpus):
    assert ray.util.client.ray.is_connected()
    from xgboost_ray.examples.simple_ray_dataset import main
    main(cpus_per_actor=1, num_actors=4)


if __name__ == "__main__":
    import pytest  # noqa: F811
    import sys
    sys.exit(pytest.main(["-v", __file__]))
