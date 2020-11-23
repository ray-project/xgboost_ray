import glob
import os

import argparse
import time

import ray
from xgboost_ray import train, RayDMatrix, RayFileType

if "OMP_NUM_THREADS" in os.environ:
    del os.environ["OMP_NUM_THREADS"]


def train_ray(num_workers, num_boost_rounds, num_files=0, use_gpu=False):
    path = "/data/parted.parquet"

    if num_files:
        files = list(sorted(glob.glob(f"{path}/**/*.parquet")))
        while num_files > len(files):
            files = files + files
        path = files[0:num_files]

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
        max_actor_restarts=2,
        num_boost_round=num_boost_rounds,
        num_actors=num_workers,
        cpus_per_actor=4,
        checkpoint_path="/tmp/checkpoint/",
        gpus_per_actor=0 if not use_gpu else 1,
        resources_per_actor={
            "actor_cpus": 4,
            "actor_gpus": 0 if not use_gpu else 1
        },
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

    args = parser.parse_args()

    num_workers = args.num_workers
    num_boost_rounds = args.num_rounds
    num_files = args.num_files
    use_gpu = args.gpu

    init_start = time.time()
    ray.init(address="auto")
    init_taken = time.time() - init_start

    full_start = time.time()
    train_taken = train_ray(num_workers, num_boost_rounds, num_files, use_gpu)
    full_taken = time.time() - full_start
    print(f"TOTAL TIME TAKEN: {full_taken:.2f} seconds "
          f"({init_taken:.2f} for init)")

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
