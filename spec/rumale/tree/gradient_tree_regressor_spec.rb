# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::GradientTreeRegressor do
  let(:two_clusters) { two_clusters_dataset }
  let(:x_bin) { two_clusters[0] }
  let(:y_bin) { Numo::DFloat.cast(two_clusters[1]) * 2 - 1 }
  let(:n_samples) { x_bin.shape[0] }
  let(:n_features) { x_bin.shape[1] }
  let(:y_pred) { 2.0 * Numo::DFloat.new(n_samples).rand - 1.0 }
  let(:grad) { -2.0 * y_bin / (1.0 + Numo::NMath.exp(2.0 * y_bin * y_pred)) }
  let(:hess) { grad.abs * (2.0 - grad.abs) }
  let(:max_depth) { nil }
  let(:max_leaf_nodes) { nil }
  let(:min_samples_leaf) { 1 }
  let(:max_features) { nil }
  let(:estimator) do
    described_class.new(reg_lambda: 1.0e-4, shrinkage_rate: 0.8,
                        max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                        min_samples_leaf: min_samples_leaf, max_features: max_features, random_seed: 1)
  end

  it 'classifies two clusters data.' do
    estimator.fit(x_bin, y_bin, grad, hess)
    expect(estimator.tree).to be_a(Rumale::Tree::Node)
    expect(estimator.leaf_weights).to be_a(Numo::DFloat)
    expect(estimator.leaf_weights).to be_contiguous
    expect(estimator.leaf_weights.shape[0]).to eq(2)
    expect(estimator.leaf_weights.shape[1]).to be_nil
    expect(estimator.feature_importances).to be_a(Numo::DFloat)
    expect(estimator.feature_importances).to be_contiguous
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(x_bin, Numo::DFloat.cast(y_bin))).to be > 0.95
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x_bin, y_bin, grad, hess)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params).to match(copied.params)
    expect(estimator.leaf_weights).to eq(copied.leaf_weights)
    expect(estimator.feature_importances).to eq(copied.feature_importances)
    expect(estimator.rng).to eq(copied.rng)
    expect(estimator.score(x_bin, y_bin)).to eq(copied.score(x_bin, y_bin))
  end

  context 'when max_depth parameter is given' do
    let(:max_depth) { 1 }

    it 'learns model with given parameters.' do
      estimator.fit(x_bin, y_bin, grad, hess)
      expect(estimator.params[:max_depth]).to eq(max_depth)
      expect(estimator.tree.left.left).to be_nil
      expect(estimator.tree.left.right).to be_nil
      expect(estimator.tree.right.left).to be_nil
      expect(estimator.tree.right.right).to be_nil
    end
  end

  context 'when max_leaf_nodes parameter is given' do
    let(:max_leaf_nodes) { 2 }

    it 'learns model with given parameters.' do
      estimator.fit(x_bin, y_bin, grad, hess)
      expect(estimator.params[:max_leaf_nodes]).to eq(max_leaf_nodes)
      expect(estimator.leaf_weights.size).to eq(max_leaf_nodes)
    end
  end
end
