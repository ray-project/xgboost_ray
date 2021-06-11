import os
import time

from xgboost_ray import train, RayDMatrix, RayParams

FILENAME_CSV = "HIGGS.csv.gz"


def download_higgs(target_file):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
          "00280/HIGGS.csv.gz"

    try:
        import urllib.request
    except ImportError as e:
        raise ValueError(
            f"Automatic downloading of the HIGGS dataset requires `urllib`."
            f"\nFIX THIS by running `pip install urllib` or manually "
            f"downloading the dataset from {url}.") from e

    print(f"Downloading HIGGS dataset to {target_file}")
    urllib.request.urlretrieve(url, target_file)
    return os.path.exists(target_file)


def main():
    # Example adapted from this blog post:
    # https://medium.com/rapids-ai/a-new-official-dask-api-for-xgboost-e8b10f3d1eb7
    # This uses the HIGGS dataset. Download here:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz

    if not os.path.exists(FILENAME_CSV):
        assert download_higgs(FILENAME_CSV), \
            "Downloading of HIGGS dataset failed."
        print("HIGGS dataset downloaded.")
    else:
        print("HIGGS dataset found locally.")

    colnames = ["label"] + ["feature-%02d" % i for i in range(1, 29)]

    dtrain = RayDMatrix(
        os.path.abspath(FILENAME_CSV), label="label", names=colnames)

    config = {
        "tree_method": "hist",
        "eval_metric": ["logloss", "error"],
    }

    evals_result = {}

    start = time.time()
    bst = train(
        config,
        dtrain,
        evals_result=evals_result,
        ray_params=RayParams(max_actor_restarts=1),
        num_boost_round=100,
        evals=[(dtrain, "train")])
    taken = time.time() - start
    print(f"TRAIN TIME TAKEN: {taken:.2f} seconds")

    bst.save_model("higgs.xgb")
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))


if __name__ == "__main__":
    import ray
    ray.init()

    start = time.time()
    main()
    taken = time.time() - start
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
