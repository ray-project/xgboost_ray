from contextlib import contextmanager
from functools import partial

import pytest
import ray

try:
    # Ray 1.3+
    from ray._private.cluster_utils import Cluster
except ImportError:
    from ray.cluster_utils import Cluster


def get_default_fixure_system_config():
    system_config = {
        "object_timeout_milliseconds": 200,
        "num_heartbeats_timeout": 10,
        "object_store_full_max_retries": 3,
        "object_store_full_initial_delay_ms": 100,
    }
    return system_config


def get_default_fixture_ray_kwargs():
    system_config = get_default_fixure_system_config()
    ray_kwargs = {
        "num_cpus": 1,
        "object_store_memory": 150 * 1024 * 1024,
        "_system_config": system_config,
    }
    return ray_kwargs


@contextmanager
def _ray_start_cluster(**kwargs):
    init_kwargs = get_default_fixture_ray_kwargs()
    num_nodes = 0
    do_init = False
    # num_nodes & do_init are not arguments for ray.init, so delete them.
    if "num_nodes" in kwargs:
        num_nodes = kwargs["num_nodes"]
        del kwargs["num_nodes"]
    if "do_init" in kwargs:
        do_init = kwargs["do_init"]
        del kwargs["do_init"]
    elif num_nodes > 0:
        do_init = True
    init_kwargs.update(kwargs)
    cluster = Cluster()
    remote_nodes = []
    for i in range(num_nodes):
        if i > 0 and "_system_config" in init_kwargs:
            del init_kwargs["_system_config"]
        remote_nodes.append(cluster.add_node(**init_kwargs))
        # We assume driver will connect to the head (first node),
        # so ray init will be invoked if do_init is true
        if len(remote_nodes) == 1 and do_init:
            ray.init(address=cluster.address)
    yield cluster
    # The code after the yield will run as teardown code.
    ray.shutdown()
    cluster.shutdown()


# This fixture will start a cluster with empty nodes.
@pytest.fixture(scope="function")
def ray_start_cluster(request):
    param = getattr(request, "param", {})
    request.cls.ray_start_cluster = partial(_ray_start_cluster, **param)
