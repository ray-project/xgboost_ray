set -e

TUNE=1

for i in "$@"
do
echo "$i"
case "$i" in
    --no-tune)
    TUNE=0
    ;;
    *)
    echo "unknown arg, $i"
    exit 1
    ;;
esac
done

pushd xgboost_ray/examples/ || exit 1
ray stop || true
echo "================"
echo "Running examples"
echo "================"
echo "running readme.py" && python readme.py
echo "running readme_sklearn_api.py" && python readme_sklearn_api.py
echo "running simple.py" && python simple.py --smoke-test
echo "running simple_predict.py" && python simple_predict.py
echo "running simple_dask.py" && python simple_dask.py --smoke-test
echo "running simple_modin.py" && python simple_modin.py --smoke-test
echo "running simple_objectstore.py" && python simple_objectstore.py --smoke-test
echo "running simple_ray_dataset.py" && python simple_objectstore.py --smoke-test
echo "running simple_partitioned.py" && python simple_partitioned.py --smoke-test

if [ "$TUNE" = "1" ]; then
  echo "running simple_tune.py" && python simple_tune.py --smoke-test
else
  echo "skipping tune example"
fi

echo "running train_on_test_data.py" && python train_on_test_data.py --smoke-test
popd

pushd xgboost_ray/tests
echo "running examples with Ray Client"
python -m pytest -v --durations=0 -x test_client.py
popd || exit 1
