if [ ! -f "./.anyscale.yaml" ]; then
  echo "Anyscale project not initialized. Please run 'anyscale init'"
  exit 1
fi

export XGBOOST_RAY_PACKAGE="git+https://github.com/krfricke/xgboost_ray@fault-tolerance#xgboost-ray"
export NUM_WORKERS=4

SESSION_NAME="xgboost_ray_release_gpu_$(date +%s)"

echo "Starting GPU cluster with ${NUM_WORKERS} worker nodes (plus the head node)"
echo "This will install xgboost_ray using the following package: ${XGBOOST_RAY_PACKAGE}"

echo Running: anyscale up --cloud-name anyscale_default_cloud --config cluster_gpu.yaml "${SESSION_NAME}"
anyscale up --cloud-name anyscale_default_cloud --config cluster_gpu.yaml "${SESSION_NAME}"
