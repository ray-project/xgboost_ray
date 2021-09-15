if [ ! -f "./.anyscale.yaml" ]; then
  echo "Anyscale project not initialized. Please run 'anyscale init'"
  exit 1
fi

NOW=$(date +%s)
export SESSION_NAME="xgboost_ray_ci_gpu_${NOW}"
export NUM_WORKERS=3
export XGBOOST_RAY_PACKAGE="git+https://github.com/ray-project/xgboost_ray.git@${GITHUB_SHA:-master}#egg=xgboost_ray[default]"
export NO_TMUX=1

./start_gpu_cluster.sh
./submit_cpu_gpu_benchmark.sh 4 100 100 --gpu --file /data/classification.parquet
anyscale down "${SESSION_NAME}"
