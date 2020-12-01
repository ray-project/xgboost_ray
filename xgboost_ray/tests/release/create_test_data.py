from examples.create_test_data import create_parquet


def main():
    create_parquet(
        "/data/parted.parquet",
        num_rows=1_500_000_000,
        num_partitions=10_000,
        num_features=4,
        num_classes=2)


if __name__ == "__main__":
    main()
