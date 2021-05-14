import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import ray

from xgboost_ray import RayDMatrix
from xgboost_ray.matrix import concat_dataframes, RayShardingMode


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
        if "sharding" not in kwargs:
            kwargs["sharding"] = RayShardingMode.BATCH
        mat = RayDMatrix(in_x, in_y, **kwargs)

        def _load_data(params):
            x = params["data"]
            y = params["label"]

            if isinstance(x, list):
                x = concat_dataframes(x)
            if isinstance(y, list):
                y = concat_dataframes(y)
            return x, y

        params = mat.get_data(rank=0, num_actors=1)
        x, y = _load_data(params)

        self.assertTrue(np.allclose(self.x, x))
        self.assertTrue(np.allclose(self.y, y))

        # Multi actor check
        mat = RayDMatrix(in_x, in_y, **kwargs)

        params = mat.get_data(rank=0, num_actors=2)
        x1, y1 = _load_data(params)

        mat.unload_data()

        params = mat.get_data(rank=1, num_actors=2)
        x2, y2 = _load_data(params)

        self.assertTrue(np.allclose(self.x, concat_dataframes([x1, x2])))
        self.assertTrue(np.allclose(self.y, concat_dataframes([y1, y2])))

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
        from xgboost_ray.data_sources.modin import MODIN_INSTALLED
        if not MODIN_INSTALLED:
            self.skipTest("Modin not installed.")
            return

        from modin.pandas import DataFrame

        in_x = DataFrame(self.x)
        in_y = DataFrame(self.y)
        self._testMatrixCreation(in_x, in_y, distributed=False)

    def testFromModinDfSeries(self):
        from xgboost_ray.data_sources.modin import MODIN_INSTALLED
        if not MODIN_INSTALLED:
            self.skipTest("Modin not installed.")
            return

        from modin.pandas import DataFrame, Series

        in_x = DataFrame(self.x)
        in_y = Series(self.y)
        self._testMatrixCreation(in_x, in_y, distributed=False)

    def testFromModinDfString(self):
        from xgboost_ray.data_sources.modin import MODIN_INSTALLED
        if not MODIN_INSTALLED:
            self.skipTest("Modin not installed.")
            return

        from modin.pandas import DataFrame

        in_df = DataFrame(self.x)
        in_df["label"] = self.y
        self._testMatrixCreation(in_df, "label", distributed=False)
        self._testMatrixCreation(in_df, "label", distributed=True)

    def testFromDaskDfSeries(self):
        from xgboost_ray.data_sources.dask import DASK_INSTALLED
        if not DASK_INSTALLED:
            self.skipTest("Dask not installed.")
            return

        import dask.dataframe as dd

        in_x = dd.from_array(self.x)
        in_y = dd.from_array(self.y)

        self._testMatrixCreation(in_x, in_y, distributed=False)

    def testFromDaskDfArray(self):
        from xgboost_ray.data_sources.dask import DASK_INSTALLED
        if not DASK_INSTALLED:
            self.skipTest("Dask not installed.")
            return

        import dask.dataframe as dd
        import dask.array as da

        in_x = dd.from_array(self.x)
        in_y = da.from_array(self.y)

        self._testMatrixCreation(in_x, in_y, distributed=False)

    def testFromDaskDfString(self):
        from xgboost_ray.data_sources.dask import DASK_INSTALLED
        if not DASK_INSTALLED:
            self.skipTest("Dask not installed.")
            return

        import dask.dataframe as dd

        in_df = dd.from_array(self.x)
        in_df["label"] = dd.from_array(self.y)

        self._testMatrixCreation(in_df, "label", distributed=False)
        self._testMatrixCreation(in_df, "label", distributed=True)

    def testFromPetastormParquetString(self):
        try:
            import petastorm  # noqa: F401
        except ImportError:
            self.skipTest("Petastorm not installed.")
            return

        with tempfile.TemporaryDirectory() as dir:
            data_file = os.path.join(dir, "data.parquet")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)
            data_df.to_parquet(data_file)

            self._testMatrixCreation(
                f"file://{data_file}", "label", distributed=False)
            self._testMatrixCreation(
                f"file://{data_file}", "label", distributed=True)

    def testFromPetastormMultiParquetString(self):
        with tempfile.TemporaryDirectory() as dir:
            data_file_1 = os.path.join(dir, "data_1.parquet")
            data_file_2 = os.path.join(dir, "data_2.parquet")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)

            df_1 = data_df[0:len(data_df) // 2]
            df_2 = data_df[len(data_df) // 2:]

            df_1.to_parquet(data_file_1)
            df_2.to_parquet(data_file_2)

            self._testMatrixCreation(
                [f"file://{data_file_1}", f"file://{data_file_2}"],
                "label",
                distributed=False)
            self._testMatrixCreation(
                [f"file://{data_file_1}", f"file://{data_file_2}"],
                "label",
                distributed=True)

    def testFromCSVString(self):
        with tempfile.TemporaryDirectory() as dir:
            data_file = os.path.join(dir, "data.csv")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)
            data_df.to_csv(data_file, header=True, index=False)

            self._testMatrixCreation(data_file, "label", distributed=False)
            with self.assertRaises(ValueError):
                self._testMatrixCreation(data_file, "label", distributed=True)

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

            self._testMatrixCreation(
                [data_file_1, data_file_2], "label", distributed=False)
            self._testMatrixCreation(
                [data_file_1, data_file_2], "label", distributed=True)

    def testFromParquetString(self):
        with tempfile.TemporaryDirectory() as dir:
            data_file = os.path.join(dir, "data.parquet")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)
            data_df.to_parquet(data_file)

            self._testMatrixCreation(data_file, "label", distributed=False)
            self._testMatrixCreation(data_file, "label", distributed=True)

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

            self._testMatrixCreation(
                [data_file_1, data_file_2], "label", distributed=False)
            self._testMatrixCreation(
                [data_file_1, data_file_2], "label", distributed=True)

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
            parquet_file = os.path.join(dir, "file.parquet")
            csv_file = os.path.join(dir, "file.csv")

            data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])
            data_df["label"] = pd.Series(self.y)

            data_df.to_parquet(parquet_file)
            data_df.to_csv(csv_file)

            mat = RayDMatrix(parquet_file, lazy=True)
            self.assertTrue(mat.distributed)

            mat = RayDMatrix(csv_file, lazy=True)
            # Single CSV files should not be distributed
            self.assertFalse(mat.distributed)

            mat = RayDMatrix([parquet_file] * 3, lazy=True)
            self.assertTrue(mat.distributed)

            mat = RayDMatrix([csv_file] * 3, lazy=True)
            self.assertTrue(mat.distributed)

            try:
                from ray.util import data as ml_data
                mat = RayDMatrix(
                    ml_data.read_parquet(parquet_file, num_shards=1),
                    lazy=True)
                self.assertTrue(mat.distributed)
            except ImportError:
                print("MLDataset not available in current Ray version. "
                      "Skipping part of test.")

    def testTooManyActorsDistributed(self):
        """Test error when too many actors are passed"""
        with self.assertRaises(RuntimeError):
            dtrain = RayDMatrix(["foo.csv"], num_actors=4, distributed=True)
            dtrain.assert_enough_shards_for_actors(4)

    def testTooManyActorsCentral(self):
        """Test error when too many actors are passed"""
        data_df = pd.DataFrame(self.x, columns=["a", "b", "c", "d"])

        with self.assertRaises(RuntimeError):
            RayDMatrix(data_df, num_actors=34, distributed=False)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
