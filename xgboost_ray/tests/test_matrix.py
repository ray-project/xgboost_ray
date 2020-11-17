import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import ray

from xgboost_ray import RayDMatrix


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

    def testSameObject(self):
        """Test that matrices are recognized as the same in an actor task."""
        ray.init(num_cpus=1)

        @ray.remote
        def same(one, two):
            return one == two

        data = RayDMatrix(self.x, self.y)
        self.assertTrue(ray.get(same.remote(data, data)))

    def _testMatrixCreation(self, in_x, in_y, **kwargs):
        mat = RayDMatrix(in_x, in_y, **kwargs)
        x, y = mat.get_data(rank=0, num_actors=1)
        print(type(x), x)
        self.assertTrue(np.allclose(self.x, x))
        print(type(y), y)
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


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
