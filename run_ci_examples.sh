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

pushd examples/ || exit 1
ray stop || true
echo "================"
echo "Running examples"
echo "================"
echo "running readme.py" && python readme.py
echo "running simple.py" && python simple.py --smoke-test
echo "running simple_predict.py" && python simple_predict.py
echo "running simple_modin.py" && python simple_modin.py --smoke-test

if [ "$TUNE" = "1" ]; then
  echo "running simple_tune.py" && python simple_tune.py --smoke-test
else
  echo "skipping tune example"
fi

echo "running train_on_test_data.py" && python train_on_test_data.py --smoke-test
popd || exit 1