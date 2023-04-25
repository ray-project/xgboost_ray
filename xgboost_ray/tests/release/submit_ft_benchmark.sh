#!/bin/bash

if [ ! -f "./.anyscale.yaml" ]; then
  echo "Anyscale project not initialized. Please run 'anyscale init'"
  exit 1
fi

ANYSCALE_CMD="python ~/xgboost_tests/benchmark_ft.py $*"

SESSION_STR=""
if [ -n "${SESSION_NAME}" ]; then
  SESSION_STR="--session-name ${SESSION_NAME}"
fi

TMUX="--tmux"
if [ "${NO_TMUX}" = "1" ]; then
  TMUX=""
fi

CMD="anyscale exec ${TMUX} ${SESSION_STR} -- ${ANYSCALE_CMD}"

echo "Running: ${CMD}"
${CMD}
