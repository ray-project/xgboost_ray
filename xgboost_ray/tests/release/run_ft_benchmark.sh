if [ ! -f "./.anyscale.yaml" ]; then
  echo "Anyscale project not initialized. Please run 'anyscale init'"
  exit 1
fi

CMD="python ~/xgboost_tests/benchmark_ft.py $*"

echo Running: anyscale exec --tmux -- "${CMD}"
anyscale exec --tmux -- "${CMD}"
