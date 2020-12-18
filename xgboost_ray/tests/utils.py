import json
import os
import tempfile
import time
from typing import Tuple, Union, List, Dict

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost.callback import TrainingCallback

from xgboost_ray.session import get_actor_rank, put_queue


def create_data(num_rows: int, num_cols: int, dtype: np.dtype = np.float32):

    return pd.DataFrame(
        np.random.uniform(0.0, 10.0, size=(num_rows, num_cols)),
        columns=[f"feature_{i}" for i in range(num_cols)],
        dtype=dtype)


def create_labels(num_rows: int,
                  num_classes: int = 2,
                  dtype: np.dtype = np.int32):

    return pd.Series(
        np.random.randint(0, num_classes, size=num_rows),
        dtype=dtype,
        name="label")


def create_parquet(filename: str,
                   num_rows: int,
                   num_features: int,
                   num_classes: int = 2,
                   num_partitions: int = 1):

    partition_rows = num_rows // num_partitions
    for partition in range(num_partitions):
        print(f"Creating partition {partition}")
        data = create_data(partition_rows, num_features)
        labels = create_labels(partition_rows, num_classes)
        partition = pd.Series(
            np.full(partition_rows, partition), dtype=np.int32)

        data["labels"] = labels
        data["partition"] = partition

        os.makedirs(filename, 0o755, exist_ok=True)
        data.to_parquet(
            filename,
            partition_cols=["partition"],
            engine="pyarrow",
            partition_filename_cb=lambda key: f"part_{key[0]}.parquet")


def create_parquet_in_tempdir(filename: str,
                              num_rows: int,
                              num_features: int,
                              num_classes: int = 2,
                              num_partitions: int = 1) -> Tuple[str, str]:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, filename)
    create_parquet(
        path,
        num_rows=num_rows,
        num_features=num_features,
        num_classes=num_classes,
        num_partitions=num_partitions)
    return temp_dir, path


def flatten_obj(obj: Union[List, Dict], keys=None, base=None):
    keys = keys or []
    base = base if base is not None else {}  # Keep same object if empty dict
    if isinstance(obj, list):
        for i, o in enumerate(obj):
            flatten_obj(o, keys + [str(i)], base)
    elif isinstance(obj, dict):
        for k, o in obj.items():
            flatten_obj(o, keys + [str(k)], base)
    else:
        base["/".join(keys)] = obj
    return base

def tree_obj(bst: xgb.Booster):
    return [json.loads(j) for j in bst.get_dump(dump_format="json")]


def _kill_callback(die_lock_file: str,
                   actor_rank: int = 0,
                   fail_iteration: int = 6):
    class _KillCallback(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if get_actor_rank() == actor_rank:
                put_queue((epoch, time.time()))
            if get_actor_rank() == actor_rank and \
                    epoch == fail_iteration and \
                    not os.path.exists(die_lock_file):

                # Get PID
                pid = os.getpid()
                print(f"Killing process: {pid}")
                with open(die_lock_file, "wt") as fp:
                    fp.write("")

                time.sleep(2)
                os.kill(pid, 9)

    return _KillCallback()


def _fail_callback(die_lock_file: str,
                   actor_rank: int = 0,
                   fail_iteration: int = 6):
    class _FailCallback(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):

            if get_actor_rank() == actor_rank:
                put_queue((epoch, time.time()))
            if get_actor_rank() == actor_rank and \
               epoch == fail_iteration and \
               not os.path.exists(die_lock_file):

                with open(die_lock_file, "wt") as fp:
                    fp.write("")
                time.sleep(2)
                import sys
                sys.exit(1)

    return _FailCallback()


def _checkpoint_callback(frequency: int = 1, before_iteration_=False):
    class _CheckpointCallback(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch % frequency == 0:
                put_queue(model.save_raw())

        def before_iteration(self, model, epoch, evals_log):
            if before_iteration_:
                self.after_iteration(model, epoch, evals_log)

    return _CheckpointCallback()
