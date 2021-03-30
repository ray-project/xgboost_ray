from typing import List, Dict

import argparse
import glob
import os

import numpy as np

import ray
from ray import tune
from ray.tune import CLIReporter
from xgboost_ray import train, RayDMatrix, RayFileType, \
    RayDeviceQuantileDMatrix, RayParams, RayShardingMode
from xgboost_ray.callback import EnvironmentCallback
from xgboost_ray.matrix import _get_sharding_indices
from xgboost_ray.tests.fault_tolerance import DelayedLoadingCallback, \
    DieCallback, FaultToleranceManager
from xgboost_ray.tests.utils import create_parquet_in_tempdir

if "OMP_NUM_THREADS" in os.environ:
    del os.environ["OMP_NUM_THREADS"]


def train_ray(train_files,
              eval_files,
              num_workers,
              num_boost_round,
              regression=False,
              use_gpu=False,
              ray_params=None,
              xgboost_params=None,
              ft_manager=None,
              aws=None,
              **kwargs):
    use_device_matrix = False
    if use_gpu:
        try:
            import cupy  # noqa: F401
            use_device_matrix = True
        except ImportError:
            use_device_matrix = False

    if use_gpu and use_device_matrix:
        dtrain = RayDeviceQuantileDMatrix(
            train_files,
            num_actors=num_workers,
            label="labels",
            ignore=["partition"],
            filetype=RayFileType.PARQUET)
        deval = RayDeviceQuantileDMatrix(
            eval_files,
            num_actors=num_workers,
            label="labels",
            ignore=["partition"],
            filetype=RayFileType.PARQUET)
    else:
        dtrain = RayDMatrix(
            train_files,
            num_actors=num_workers,
            label="labels",
            ignore=["partition"],
            filetype=RayFileType.PARQUET)
        deval = RayDMatrix(
            eval_files,
            num_actors=num_workers,
            label="labels",
            ignore=["partition"],
            filetype=RayFileType.PARQUET)

    config = xgboost_params or {"tree_method": "hist"}

    if use_gpu:
        config.update({"tree_method": "gpu_hist"})

    if not regression:
        # Classification
        config.update({
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
        })
        return_metric = "error"
    else:
        # Regression
        config.update({
            "objective": "reg:squarederror",
            "eval_metric": ["logloss", "rmse"],
        })
        return_metric = "rmse"

    xgboost_callbacks = []
    distributed_callbacks = []
    if ft_manager:
        delay_callback = DelayedLoadingCallback(
            ft_manager, reload_data=True, sleep_time=0.1)
        distributed_callbacks.append(delay_callback)

        die_callback = DieCallback(ft_manager, training_delay=0.1)
        xgboost_callbacks.append(die_callback)

    if aws:
        aws_callback = EnvironmentCallback(aws)
        distributed_callbacks.append(aws_callback)
        os.environ.update(aws)

    ray_params = ray_params or RayParams()
    ray_params.num_actors = num_workers
    ray_params.gpus_per_actor = 0 if not use_gpu else 1
    ray_params.distributed_callbacks = distributed_callbacks

    evals_result = {}
    additional_results = {}
    bst = train(
        config,
        dtrain,
        evals_result=evals_result,
        additional_results=additional_results,
        num_boost_round=num_boost_round,
        ray_params=ray_params,
        evals=[(dtrain, "train"), (deval, "eval")],
        callbacks=xgboost_callbacks,
        **kwargs)

    bst.save_model("benchmark_{}.xgb".format("cpu" if not use_gpu else "gpu"))
    print("Final training error: {:.4f}".format(
        evals_result["train"][return_metric][-1]))

    results = {
        "train-logloss": evals_result["train"]["logloss"][-1],
        f"train-{return_metric}": evals_result["train"][return_metric][-1],
        "eval-logloss": evals_result["eval"]["logloss"][-1],
        f"eval-{return_metric}": evals_result["eval"][return_metric][-1],
        "total_n": additional_results["total_n"]
    }

    return bst, results


def ft_setup(workers: List[int], num_rounds: int, die_round_factor: 0.25,
             comeback_round_factor: 0.75):
    """Setup fault tolerance manager, schedule kills and comebacks"""
    if workers is None:
        return None

    ft_manager = FaultToleranceManager.remote()

    # Choose some nodes to kill
    die_round = int(die_round_factor * num_rounds)
    comeback_round = int(comeback_round_factor * num_rounds)

    for worker in workers:
        ft_manager.schedule_kill.remote(rank=worker, boost_round=die_round)
        ft_manager.delay_return.remote(
            rank=1,
            start_boost_round=die_round - 2,
            end_boost_round=comeback_round - 1)

    print(f"Scheduled workers {list(workers)} to die at round {die_round} "
          f"and to come back at round {comeback_round} "
          f"(total {num_rounds} training rounds)")

    return ft_manager


def run_experiments(config, files, aws):
    """Ray Tune-compatible function trainable to run experiments"""
    os.environ["RXGB_ALLOW_ELASTIC_TUNE"] = "1"

    condition = config["condition"]

    num_boost_round = config["num_boost_round"]
    num_workers = config["num_workers"]
    num_affected_workers = config["affected_workers"]
    regression = config["regression"]
    use_gpu = config["use_gpu"]
    seed = config["seed"]

    metric = "eval-error" if not regression else "eval-rmse"

    xgboost_params: Dict = config["xgboost_params"]
    ray_params: RayParams = config["ray_params"]

    # Select a fixed subset of the files for evaluation only
    np.random.seed(seed)
    np.random.shuffle(files)
    last_train_index = int(0.8 * len(files))
    train_files = list(files[:last_train_index])
    eval_files = list(files[last_train_index:])

    if num_affected_workers:
        affected_workers = np.random.choice(
            np.arange(1, num_workers),
            size=num_affected_workers,
            replace=False).tolist()
    else:
        affected_workers = None

    np.random.seed(seed)  # Re-seed because of conditional evaluation

    # Dataset to train on
    sharding_mode = RayShardingMode.INTERLEAVED

    if condition == "calibrate":
        final_files = train_files
        final_workers = num_workers

        ray_params.num_actors = final_workers

        bst, results = train_ray(
            train_files=final_files,
            eval_files=eval_files,
            num_workers=final_workers,
            num_boost_round=num_boost_round,
            regression=regression,
            use_gpu=use_gpu,
            ray_params=ray_params,
            xgboost_params=xgboost_params,
            ft_manager=None,
            aws=aws,
            early_stopping_rounds=10)

        return results

    if condition == "fewer_workers":
        # Sanity check: Just train with fewer workers
        remove_shards = []

        if affected_workers is not None:
            for rank in affected_workers:
                remove_shards += _get_sharding_indices(
                    sharding=sharding_mode,
                    rank=rank,
                    num_actors=num_workers,
                    n=len(train_files))

            mask = np.ones(len(train_files), dtype=bool)
            mask[remove_shards] = False

            final_files = np.array(train_files)[mask].tolist()
            final_workers = num_workers - len(affected_workers)
        else:
            final_files = train_files
            final_workers = num_workers

        ray_params.num_actors = final_workers

        bst, results = train_ray(
            train_files=final_files,
            eval_files=eval_files,
            num_workers=final_workers,
            num_boost_round=num_boost_round,
            regression=regression,
            use_gpu=use_gpu,
            ray_params=ray_params,
            xgboost_params=xgboost_params,
            ft_manager=None,
            aws=aws)

        return results

    if num_affected_workers == 0:
        # No duplicate baseline runs
        return {metric: float("inf")}

    if condition == "non_elastic":
        # Non-elastic training: Actors die after 50% and come back
        ray_params.elastic_training = False
        ray_params.max_failed_actors = len(affected_workers)
        ray_params.max_actor_restarts = 1

        ft_manager = ft_setup(
            workers=affected_workers,
            num_rounds=num_boost_round,
            die_round_factor=0.5,
            comeback_round_factor=0.0,
        )

    elif condition == "elastic_no_comeback":
        # Elastic training: Actors die after 50% and don't come back
        ray_params.elastic_training = True
        ray_params.max_failed_actors = len(affected_workers)
        ray_params.max_actor_restarts = 1

        ft_manager = ft_setup(
            workers=affected_workers,
            num_rounds=num_boost_round,
            die_round_factor=0.5,
            comeback_round_factor=1.1,
        )

    elif condition == "elastic_comeback":
        # Elastic training: Actors die after 50% and come back
        ray_params.elastic_training = True
        ray_params.max_failed_actors = len(affected_workers)
        ray_params.max_actor_restarts = 1

        ft_manager = ft_setup(
            workers=affected_workers,
            num_rounds=num_boost_round,
            die_round_factor=0.5,
            comeback_round_factor=0.75,
        )
    else:
        raise ValueError("Unknown condition:", condition)

    bst, results = train_ray(
        train_files=train_files,
        eval_files=eval_files,
        num_workers=num_workers,
        num_boost_round=num_boost_round,
        regression=regression,
        use_gpu=use_gpu,
        ray_params=ray_params,
        xgboost_params=xgboost_params,
        ft_manager=ft_manager,
        aws=aws)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("num_workers", type=int, help="num workers")
    parser.add_argument("num_rounds", type=int, help="num boost rounds")
    parser.add_argument("num_files", type=int, help="num files")

    parser.add_argument(
        "--cpu", default=0, type=int, help="num cpus per worker")

    parser.add_argument(
        "--file", default="/data/parted.parquet", type=str, help="data file")

    parser.add_argument(
        "--regression", action="store_true", default=False, help="regression")

    parser.add_argument(
        "--gpu", action="store_true", default=False, help="gpu")

    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="calibrate boost rounds")

    parser.add_argument(
        "--smoke-test", action="store_true", default=False, help="smoke test")

    args = parser.parse_args()

    num_workers = args.num_workers
    num_boost_round = args.num_rounds
    num_files = args.num_files
    use_gpu = args.gpu

    aws = None

    temp_dir = None
    if args.smoke_test:
        temp_dir, files = create_parquet_in_tempdir(
            filename="smoketest.parquet",
            num_rows=args.num_workers * 500,
            num_features=4,
            num_classes=2,
            num_partitions=args.num_workers * 10)
        use_gpu = False
    else:
        path = args.file
        if path.startswith("s3://"):
            base, num_partitions = path.split("#", maxsplit=1)
            num_partitions = int(num_partitions)
            files = [
                f"{base}/partition={i}/part_{i}.parquet"
                for i in range(num_partitions)
            ]
            print(f"Using S3 dataset with base {base} and "
                  f"{num_partitions} partitions.")
            try:
                aws = {
                    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                    "AWS_SECRET_ACCESS_KEY": os.environ[
                        "AWS_SECRET_ACCESS_KEY"],
                    "AWS_SESSION_TOKEN": os.environ["AWS_SESSION_TOKEN"],
                }
            except KeyError as e:
                raise ValueError(
                    "Trying to access AWS S3, but credentials are not set "
                    "in the environment. Did you forget to set your "
                    "credentials?") from e

        elif not os.path.exists(path):
            raise ValueError(
                f"Benchmarking data not found: {path}."
                f"\nFIX THIS by running `python create_test_data.py` first.")
        else:
            files = sorted(glob.glob(f"{path}/**/*.parquet"))
            print(f"Using local dataset with base {path} and "
                  f"{len(files)} partitions.")

    if num_files:
        while num_files > len(files):
            files = files + files
        files = files[0:num_files]

    if args.smoke_test:
        ray.init(num_cpus=num_workers)
    else:
        ray.init(address="auto")

    ray_params = RayParams(
        num_actors=num_workers,
        cpus_per_actor=args.cpu,
        checkpoint_frequency=1,
    )

    config = {
        "num_workers": num_workers,
        "num_boost_round": num_boost_round,
        "seed": 1000,
        "condition": tune.grid_search([
            "fewer_workers",
            "non_elastic",
            "elastic_no_comeback",
            "elastic_comeback",
        ]),
        "affected_workers": tune.grid_search([0, 1, 2, 3]),
        "regression": args.regression,
        "use_gpu": args.gpu,
        "xgboost_params": {},
        "ray_params": ray_params,
    }

    if args.calibrate:
        config["condition"] = "calibrate"
        config["affected_workers"] = 0

    metric = "eval-error" if not args.regression else "eval-rmse"
    train_metric = "train-error" if not args.regression else "train-rmse"

    reporter = CLIReporter(
        parameter_columns=["condition", "affected_workers"],
        metric_columns=[
            metric, "eval-logloss", train_metric, "total_n", "time_total_s"
        ],
        print_intermediate_tables=True)

    analysis = tune.run(
        tune.with_parameters(run_experiments, files=files, aws=aws),
        config=config,
        metric=metric,
        mode="min",
        resources_per_trial=ray_params.get_tune_resources(),
        reuse_actors=True,
        progress_reporter=reporter,
        log_to_file=True,
        verbose=2)

    print(f"Best config: {analysis.best_config} "
          f"with result {analysis.best_result}")
