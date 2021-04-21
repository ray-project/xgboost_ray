if [ ! -f "./.anyscale.yaml" ]; then
  echo "Anyscale project not initialized. Please run 'anyscale init'"
  exit 1
fi

ANYSCALE_CMD="python ~/xgboost_tests/benchmark_cpu_gpu.py $*"

SESSION_STR=""
if [ -n "${SESSION_NAME}" ]; then
  SESSION_STR="--session-name ${SESSION_NAME}"
fi

CMD="anyscale exec --tmux ${SESSION_STR} -- ${ANYSCALE_CMD}"

echo "Running: ${CMD}"
${CMD}
