if [ ! -f "./.anyscale.yaml" ]; then
  echo "Anyscale project not initialized. Please run 'anyscale init'"
  exit 1
fi

export XGBOOST_RAY_PACKAGE="git+https://github.com/krfricke/xgboost_ray@fault-tolerance#xgboost-ray"
export NUM_WORKERS=4

SESSION_NAME=${SESSION_NAME:-xgboost_ray_release_ft_$(date +%s)}

echo "Starting FT cluster with ${NUM_WORKERS} worker nodes (plus the head node)"
echo "This will install xgboost_ray using the following package: ${XGBOOST_RAY_PACKAGE}"

CMD="anyscale up --cloud-name anyscale_default_cloud --config cluster_ft.yaml ${SESSION_NAME}"

echo "Running: ${CMD}"
${CMD}
