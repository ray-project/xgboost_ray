if [ ! -f "./.anyscale.yaml" ]; then
  echo "Anyscale project not initialized. Please run 'anyscale init'"
  exit 1
fi

export XGBOOST_RAY_PACKAGE="${XGBOOST_RAY_PACKAGE:-xgboost_ray}"

SESSION_NAME=${SESSION_NAME:-xgboost_ray_release_ft_$(date +%s)}

echo "Starting FT cluster"
echo "This will install xgboost_ray using the following package: ${XGBOOST_RAY_PACKAGE}"

CMD="anyscale up --cloud-name anyscale_default_cloud --config cluster_ft.yaml ${SESSION_NAME}"

echo "Running: ${CMD}"
${CMD}
