import glob
import os

import argparse
import shutil
import time

import ray
from xgboost_ray import train, RayDMatrix, RayFileType, \
    RayDeviceQuantileDMatrix, RayParams
from xgboost_ray.tests.utils import create_parquet_in_tempdir

if "OMP_NUM_THREADS" in os.environ:
    del os.environ["OMP_NUM_THREADS"]


def train_ray(path,
              num_workers,
              num_boost_rounds,
              num_files=0,
              use_gpu=False,
              smoke_test=False):
    if num_files:
        files = sorted(glob.glob(f"{path}/**/*.parquet"))
        while num_files > len(files):
            files = files + files
        path = files[0:num_files]

    use_device_matrix = False
    if use_gpu:
        try:
            import cupy  # noqa: F401
            use_device_matrix = True
        except ImportError:
            use_device_matrix = False

    if use_device_matrix:
        dtrain = RayDeviceQuantileDMatrix(
            path,
            num_actors=num_workers,
            label="labels",
            ignore=["partition"],
            filetype=RayFileType.PARQUET)
    else:
        dtrain = RayDMatrix(
            path,
            num_actors=num_workers,
            label="labels",
            ignore=["partition"],
            filetype=RayFileType.PARQUET)

    config = {
        "tree_method": "hist" if not use_gpu else "gpu_hist",
        "eval_metric": ["logloss", "error"],
    }

    start = time.time()
    evals_result = {}
    bst = train(
        config,
        dtrain,
        evals_result=evals_result,
        num_boost_round=num_boost_rounds,
        ray_params=RayParams(
            max_actor_restarts=2,
            num_actors=num_workers,
            cpus_per_actor=4,
            checkpoint_path="/tmp/checkpoint/",
            gpus_per_actor=0 if not use_gpu else 1,
            resources_per_actor={
                "actor_cpus": 4 if not smoke_test else 1,
                "actor_gpus": 0 if not use_gpu else 1
            } if not smoke_test else None),
        evals=[(dtrain, "train")])
    taken = time.time() - start
    print(f"TRAIN TIME TAKEN: {taken:.2f} seconds")

    bst.save_model("benchmark_{}.xgb".format("cpu" if not use_gpu else "gpu"))
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))
    return taken


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("num_workers", type=int, help="num workers")
    parser.add_argument("num_rounds", type=int, help="num boost rounds")
    parser.add_argument("num_files", type=int, help="num files")

    parser.add_argument(
        "--gpu", action="store_true", default=False, help="gpu")

    parser.add_argument(
        "--smoke-test", action="store_true", default=False, help="gpu")

    args = parser.parse_args()

    num_workers = args.num_workers
    num_boost_rounds = args.num_rounds
    num_files = args.num_files
    use_gpu = args.gpu

    temp_dir = None
    if args.smoke_test:
        temp_dir, path = create_parquet_in_tempdir(
            filename="smoketest.parquet",
            num_rows=args.num_workers * 500,
            num_features=4,
            num_classes=2,
            num_partitions=args.num_workers * 10)
        use_gpu = False
    else:
        path = "/data/parted.parquet"
        if not os.path.exists(path):
            raise ValueError(
                f"Benchmarking data not found: {path}."
                f"\nFIX THIS by running `python create_test_data.py` first.")

    init_start = time.time()
    ray.init()
    init_taken = time.time() - init_start

    full_start = time.time()
    train_taken = train_ray(path, num_workers, num_boost_rounds, num_files,
                            use_gpu, args.smoke_test)
    full_taken = time.time() - full_start
    print(f"TOTAL TIME TAKEN: {full_taken:.2f} seconds "
          f"({init_taken:.2f} for init)")

    if args.smoke_test:
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        with open("res.csv", "at") as fp:
            fp.writelines([
                ",".join([
                    str(e) for e in [
                        num_workers, num_files,
                        int(use_gpu), num_boost_rounds, init_taken, full_taken,
                        train_taken
                    ]
                ]) + "\n"
            ])
