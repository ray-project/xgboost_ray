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

pushd xgboost_ray/tests || exit 1
echo "============="
echo "Running tests"
echo "============="
END_STATUS=0
if ! python -m pytest -vv -s --log-cli-level=DEBUG --durations=0 -x "test_colocation.py" ; then END_STATUS=1; fi
if ! python -m pytest -v --durations=0 -x "test_matrix.py" ; then END_STATUS=1; fi
if ! python -m pytest -v --durations=0 -x "test_data_source.py" ; then END_STATUS=1; fi
if ! python -m pytest -v --durations=0 -x "test_xgboost_api.py" ; then END_STATUS=1; fi
if ! python -m pytest -v --durations=0 -x "test_fault_tolerance.py" ; then END_STATUS=1; fi
if ! python -m pytest -v --durations=0 -x "test_end_to_end.py" ; then END_STATUS=1; fi
if ! python -m pytest -v --durations=0 -x "test_sklearn.py" ; then END_STATUS=1; fi
if ! python -m pytest -v --durations=0 -x "test_sklearn_matrix.py" ; then END_STATUS=1; fi

if [ "$TUNE" = "1" ]; then
  if ! python -m pytest -v --durations=0 -x "test_tune.py" ; then END_STATUS=1; fi
else
  echo "skipping tune tests"
fi

echo "running smoke test on benchmark_cpu_gpu.py" && if ! python release/benchmark_cpu_gpu.py 2 10 20 --smoke-test; then END_STATUS=1; fi
popd || exit 1

if [ "$END_STATUS" = "1" ]; then
  echo "At least one test has failed, exiting with code 1"
fi
exit "$END_STATUS"