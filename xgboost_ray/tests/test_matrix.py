import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import ray

from xgboost_ray import RayDMatrix
from xgboost_ray.matrix import concat_dataframes


class XGBoostRayDMatrixTest(unittest.TestCase):
    """This test suite validates core RayDMatrix functionality."""

    def setUp(self):
        repeat = 8  # Repeat data a couple of times for stability
        self.x = np.array([
            [1, 0, 0, 0],  # Feature 0 -> Label 0
            [0, 1, 0, 0],  # Feature 1 -> Label 1
            [0, 0, 1, 1],  # Feature 2+3 -> Label 2
            [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
        ] * repeat)
        self.y = np.array([0, 1, 2, 3] * repeat)

    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=1, local_mode=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def testSameObject(self):
        """Test that matrices are recognized as the same in an actor task."""

        @ray.remote
        def same(one, two):
            return one == two

        data = RayDMatrix(self.x, self.y)
        self.assertTrue(ray.get(same.remote(data, data)))

    def _testMatrixCreation(self, in_x, in_y, **kwargs):
        mat = RayDMatrix(in_x, in_y, **kwargs)
        params = mat.get_data(rank=0, num_actors=1)

        x = params["data"]
        y = params["label"]

        if isinstance(x, list):
            x = concat_dataframes(x)
        if isinstance(y, list):
            y = concat_dataframes(y)

        self.assertTrue(np.allclose(self.x, x))
        self.assertTrue(np.allclose(self.y, y))

    def testFromNumpy(self):
        in_x = self.x
        in_y = self.y
        self._testMatrixCreation(in_x, in_y)

    def testFromPandasDfDf(self):
        in_x = pd.DataFrame(self.x)
        in_y = pd.DataFrame(self.y)
        self._testMatrixCreation(in_x, in_y)

    def testFromPandasDfSeries(self):
        in_x = pd.DataFrame(self.x)
        in_y = pd.Series(self.y)
        self._testMatrixCreation(in_x, in_y)

    def testFromPandasDfString(self):
        in_df = pd.DataFrame(self.x)
        in_df["label"] = self.y
        self._testMatrixCreation(in_df, "label")

    def testFromModinDfDf(self):
        try:
            from modin.pandas import DataFrame
        except ImportError:
            self.skipTest("Modin not installed.")
            return

        in_x = DataFrame(self.x)
        in_y = DataFrame(self.y)
        self._testMatrixCreation(in_x, in_y)

    def testFromModinDfSeries(self):
        try:
            from modin.pandas import DataFrame, Series
        except ImportError:
            self.skipTest("Modin not installed.")
            return

        in_x = DataFrame(self.x)
        in_y = Series(self.y)
        self._testMatrixCreation(in_x, in_y)

    def testFromModinDfString(self):
        try:
            from modin.pandas import DataFrame
        except ImportError:
            self.skipTest("Modin not installed.")
            return

        in_df = DataFrame(self.x)
        in_df["label"] = self.y
        self._testMatrixCreation(in_df, "label")

    def testFromCSVString(self):
        with tempfile.TemporaryDirectory() as dir:
            data_file = os.path.join(dir, "data.csv")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)
            data_df.to_csv(data_file, header=True, index=False)

            self._testMatrixCreation(data_file, "label")

    def testFromMultiCSVString(self):
        with tempfile.TemporaryDirectory() as dir:
            data_file_1 = os.path.join(dir, "data_1.csv")
            data_file_2 = os.path.join(dir, "data_2.csv")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)

            df_1 = data_df[0:len(data_df) // 2]
            df_2 = data_df[len(data_df) // 2:]

            df_1.to_csv(data_file_1, header=True, index=False)
            df_2.to_csv(data_file_2, header=True, index=False)

            self._testMatrixCreation([data_file_1, data_file_2], "label")

    def testFromParquetString(self):
        with tempfile.TemporaryDirectory() as dir:
            data_file = os.path.join(dir, "data.parquet")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)
            data_df.to_parquet(data_file)

            self._testMatrixCreation(data_file, "label")

    def testFromMultiParquetString(self):
        with tempfile.TemporaryDirectory() as dir:
            data_file_1 = os.path.join(dir, "data_1.parquet")
            data_file_2 = os.path.join(dir, "data_2.parquet")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)

            df_1 = data_df[0:len(data_df) // 2]
            df_2 = data_df[len(data_df) // 2:]

            df_1.to_parquet(data_file_1)
            df_2.to_parquet(data_file_2)

            self._testMatrixCreation([data_file_1, data_file_2], "label")

    def testFromMLDataset(self):
        try:
            from ray.util import data as ml_data
        except ImportError:
            self.skipTest("MLDataset not available in current Ray version.")
            return

        with tempfile.TemporaryDirectory() as dir:
            data_file_1 = os.path.join(dir, "data_1.parquet")
            data_file_2 = os.path.join(dir, "data_2.parquet")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)

            df_1 = data_df[0:len(data_df) // 2]
            df_2 = data_df[len(data_df) // 2:]

            df_1.to_parquet(data_file_1)
            df_2.to_parquet(data_file_2)

            dataset = ml_data.read_parquet(
                [data_file_1, data_file_2], num_shards=2)

            self._testMatrixCreation(dataset, "label", distributed=False)
            self._testMatrixCreation(dataset, "label", distributed=True)

    def testDetectDistributed(self):
        with tempfile.TemporaryDirectory() as dir:
            data_file = os.path.join(dir, "file.parquet")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)

            data_df.to_parquet(data_file)

            mat = RayDMatrix(data_file, lazy=True)
            self.assertTrue(mat.distributed)

            mat = RayDMatrix([data_file] * 3, lazy=True)
            self.assertTrue(mat.distributed)

            try:
                from ray.util import data as ml_data
                mat = RayDMatrix(
                    ml_data.read_parquet(data_file, num_shards=1), lazy=True)
                self.assertTrue(mat.distributed)
            except ImportError:
                print("MLDataset not available in current Ray version. "
                      "Skipping part of test.")


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
