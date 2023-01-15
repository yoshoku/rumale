#!/bin/bash

BASE_DIR=`pwd`
GEMS=("rumale-core" "rumale-evaluation_measure" "rumale-preprocessing" \
"rumale-clustering" "rumale-decomposition" "rumale-tree" "rumale-linear_model" \
"rumale-feature_extraction" "rumale-kernel_approximation" "rumale-kernel_machine" \
"rumale-manifold" "rumale-metric_learning" "rumale-naive_bayes" "rumale-nearest_neighbors" \
"rumale-neural_network" "rumale-model_selection" "rumale-pipeline"  "rumale-ensemble" "rumale")

for GEM in "${GEMS[@]}"; do
  cd "${BASE_DIR}/${GEM}"
  bundle install --quiet --jobs 4 --retry 3
  bundle exec rake || exit 1
done
