"""
Test Ray Tune trial placement across cluster nodes.

Example: Run this script on a cluster with 4 workers nodes a 4 CPUs.

    ray up -y tune_cluster.yaml

    ray attach tune_cluster.yaml

    python /release_tests/tune_placement.py 4 4 10 10 --fake-data

This starts 4 trials à 4 actors training 10 boost rounds on 10 data
partitions per actor. This will use fake data created before training.

This test will then confirm that actors of the same trial are PACKed
on the same nodes. In practice we check that each node IP address only
hosts actors of the same Ray Tune trial.
"""

import json
import os

import argparse
import shutil
import time
from collections import defaultdict

from xgboost_ray.compat import TrainingCallback

import ray

from ray import tune
from ray.tune.session import get_trial_id
from ray.tune.integration.docker import DockerSyncer
from ray.util import get_node_ip_address

from benchmark_cpu_gpu import train_ray
from xgboost_ray import RayParams
from xgboost_ray.session import put_queue
from xgboost_ray.tests.utils import create_parquet
from xgboost_ray.tune import TuneReportCallback

if "OMP_NUM_THREADS" in os.environ:
    del os.environ["OMP_NUM_THREADS"]


class PlacementCallback(TrainingCallback):
    """This callback collects the Ray Tune trial ID and node IP"""

    def before_training(self, model):
        ip_address = get_node_ip_address()
        put_queue(ip_address)
        return model

    def after_iteration(self, model, epoch, evals_log):
        if epoch == 1:
            time.sleep(2)
        elif epoch == 2:
            time.sleep(8)


def tune_test(path,
              num_trials,
              num_workers,
              num_boost_rounds,
              num_files=0,
              regression=False,
              use_gpu=False,
              fake_data=False,
              smoke_test=False):
    ray_params = RayParams(
        elastic_training=False,
        max_actor_restarts=0,
        num_actors=num_workers,
        cpus_per_actor=1,
        gpus_per_actor=0 if not use_gpu else 1)

    def local_train(config):
        temp_dir = None
        if fake_data or smoke_test:
            temp_dir = "/tmp/release_test_data"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            os.makedirs(temp_dir, 0o755)
            local_path = os.path.join(temp_dir, "smoketest.parquet")

            create_parquet(
                filename=local_path,
                num_rows=args.num_workers * 500,
                num_features=4,
                num_classes=2,
                num_partitions=args.num_workers * 10)
        else:
            if not os.path.exists(path):
                raise ValueError(
                    f"Benchmarking data not found: {path}."
                    f"\nFIX THIS by running `python create_test_data.py` "
                    f"on all nodes first.")
            local_path = path

        xgboost_params = {
            "tree_method": "hist" if not use_gpu else "gpu_hist",
        }

        xgboost_params.update({
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
        })

        xgboost_params.update(config)

        additional_results = {}

        bst, time_taken = train_ray(
            path=local_path,
            num_workers=num_workers,
            num_boost_rounds=num_boost_rounds,
            num_files=num_files,
            regression=regression,
            use_gpu=use_gpu,
            smoke_test=smoke_test,
            ray_params=ray_params,
            xgboost_params=xgboost_params,
            # kwargs
            additional_results=additional_results,
            callbacks=[PlacementCallback(),
                       TuneReportCallback()])

        bst.save_model("tuned.xgb")

        trial_ips = []
        for rank, ips in enumerate(additional_results["callback_returns"]):
            for ip in ips:
                trial_ips.append(ip)

        tune_trial = get_trial_id()
        with tune.checkpoint_dir(num_boost_rounds + 1) as checkpoint_dir:
            with open(
                    os.path.join(checkpoint_dir, "callback_returns.json"),
                    "wt") as f:
                json.dump({tune_trial: trial_ips}, f)

        if temp_dir:
            shutil.rmtree(temp_dir)

    search_space = {
        "eta": tune.loguniform(1e-4, 1e-1),
        "subsample": tune.uniform(0.5, 1.0),
        "max_depth": tune.randint(1, 9)
    }

    analysis = tune.run(
        local_train,
        config=search_space,
        num_samples=num_trials,
        sync_config=tune.SyncConfig(sync_to_driver=DockerSyncer),
        resources_per_trial=ray_params.get_tune_resources())

    # In our PACK scheduling, we expect that each IP hosts only workers
    # for one Ray Tune trial.
    ip_to_trials = defaultdict(list)
    for trial in analysis.trials:
        trial = trial
        with open(
                os.path.join(trial.checkpoint.value, "callback_returns.json"),
                "rt") as f:
            trial_to_ips = json.load(f)
        for tune_trial, ips in trial_to_ips.items():
            for node_ip in ips:
                ip_to_trials[node_ip].append(tune_trial)

    fail = False
    for ip, trial_ids in ip_to_trials.items():
        print(f"For IP {ip} got trial IDs {trial_ids}")
        fail = fail or any(trial_id != trial_ids[0] for trial_id in trial_ids)

    if fail:
        raise ValueError("Different trial IDs found on same node.")
    else:
        print("Success.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ray Tune placement "
                                     "strategy")

    parser.add_argument("num_trials", type=int, help="num trials")
    parser.add_argument(
        "num_workers", type=int, help="num workers (per trial)")
    parser.add_argument("num_rounds", type=int, help="num boost rounds")
    parser.add_argument("num_files", type=int, help="num files (per trial)")

    parser.add_argument(
        "--file", default="/data/parted.parquet", type=str, help="data file")

    parser.add_argument(
        "--regression", action="store_true", default=False, help="regression")

    parser.add_argument(
        "--gpu", action="store_true", default=False, help="gpu")

    parser.add_argument(
        "--fake-data", action="store_true", default=False, help="fake data")

    parser.add_argument(
        "--smoke-test", action="store_true", default=False, help="smoke test")

    args = parser.parse_args()

    num_trials = args.num_trials
    num_workers = args.num_workers
    num_boost_rounds = args.num_rounds
    num_files = args.num_files
    use_gpu = args.gpu

    if args.smoke_test:
        use_gpu = False

    init_start = time.time()
    if args.smoke_test:
        ray.init(num_cpus=num_workers)
    else:
        ray.init(address="auto")

    full_start = time.time()
    tune_test(
        path=args.file,
        num_trials=num_trials,
        num_workers=num_workers,
        num_boost_rounds=num_boost_rounds,
        num_files=num_files,
        regression=args.regression,
        use_gpu=use_gpu,
        fake_data=args.fake_data,
        smoke_test=args.smoke_test)
    full_taken = time.time() - full_start
    print(f"TOTAL TIME TAKEN: {full_taken:.2f} seconds ")
