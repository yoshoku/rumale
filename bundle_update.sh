#!/bin/zsh

BASE_DIR=`pwd`
GEMS=("rumale-core" "rumale-evaluation_measure" "rumale-preprocessing" "rumale-model_selection" \
"rumale-pipeline" "rumale-clustering" "rumale-decomposition" "rumale-tree" "rumale-ensemble" \
"rumale-feature_extraction" "rumale-kernel_approximation" "rumale-kernel_machine" "rumale-linear_model" \
"rumale-manifold" "rumale-metric_learning" "rumale-naive_bayes" "rumale-nearest_neighbors" \
"rumale-neural_network" "rumale")

for GEM in "${GEMS[@]}"; do
  cd "${BASE_DIR}/${GEM}"
  bundle update
done
