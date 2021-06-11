from xgboost_ray.tests.utils import create_parquet


def main():
    create_parquet(
        "example.parquet",
        num_rows=1_000_000,
        num_partitions=100,
        num_features=8,
        num_classes=2)


if __name__ == "__main__":
    main()
