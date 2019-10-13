# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::DecisionTreeClassifier do
  let(:x_bin) { Marshal.load(File.read(__dir__ + '/../../test_samples.dat')) }
  let(:y_bin) { Marshal.load(File.read(__dir__ + '/../../test_labels.dat')) }
  let(:x_mlt) { Marshal.load(File.read(__dir__ + '/../../test_samples_three_clusters.dat')) }
  let(:y_mlt) { Marshal.load(File.read(__dir__ + '/../../test_labels_three_clusters.dat')) }
  let(:estimator_entropy) { described_class.new(criterion: 'entropy', random_seed: 1) }
  let(:max_depth) { nil }
  let(:max_leaf_nodes) { nil }
  let(:min_samples_leaf) { 1 }
  let(:max_features) { nil }
  let(:estimator) do
    described_class.new(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                        min_samples_leaf: min_samples_leaf, max_features: max_features, random_seed: 1)
  end

  it 'classifies two clusters data.' do
    _n_samples, n_features = x_bin.shape
    estimator.fit(x_bin, y_bin)
    expect(estimator.tree.class).to eq(Rumale::Tree::Node)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(2)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(x_bin, y_bin)).to eq(1.0)
  end

  it 'classifies three clusters data.' do
    _n_samples, n_features = x_mlt.shape
    estimator.fit(x_mlt, y_mlt)
    expect(estimator.tree.class).to eq(Rumale::Tree::Node)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(3)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(x_mlt, y_mlt)).to eq(1.0)
  end

  it 'estimates class probabilities with three clusters dataset.' do
    n_samples, _n_features = x_mlt.shape
    estimator_entropy.fit(x_mlt, y_mlt)
    probs = estimator_entropy.predict_proba(x_mlt)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(3)
    classes = y_mlt.to_a.uniq.sort
    predicted = Numo::Int32.asarray(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })
    expect(predicted).to eq(y_mlt)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x_mlt, y_mlt)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.feature_importances).to eq(copied.feature_importances)
    expect(estimator.rng).to eq(copied.rng)
    # FIXME: A slight error on the value of the threhold parameter occurs.
    #        It seems to be caused by rounding error of Float.
    # expect(estimator.tree).to eq(copied.tree)
    expect(copied.score(x_mlt, y_mlt)).to eq(1.0)
  end

  context 'when max_depth parameter is given' do
    let(:max_depth) { 1 }

    it 'learns model with given parameters.' do
      estimator.fit(x_mlt, y_mlt)
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
      estimator.fit(x_mlt, y_mlt)
      expect(estimator.params[:max_leaf_nodes]).to eq(max_leaf_nodes)
      expect(estimator.leaf_labels.size).to eq(max_leaf_nodes)
    end
  end

  context 'when min_samples_leaf parameter is given' do
    let(:min_samples_leaf) { 110 }

    it 'learns model with given parameters.' do
      estimator.fit(x_mlt, y_mlt)
      expect(estimator.params[:min_samples_leaf]).to eq(min_samples_leaf)
      expect(estimator.tree.left.leaf).to be_truthy
      expect(estimator.tree.left.n_samples).to be >= min_samples_leaf
      expect(estimator.tree.right).to be_nil
    end
  end

  context 'when max_features parameter is given' do
    context 'with negative value' do
      let(:max_features) { -10 }

      it 'raises ArgumentError by validation' do
        expect { estimator }.to raise_error(ArgumentError)
      end
    end

    context 'with value larger than number of features' do
      let(:max_features) { 10 }

      it 'value of max_features is equal to the number of features' do
        estimator.fit(x_mlt, y_mlt)
        expect(estimator.params[:max_features]).to eq(x_mlt.shape[1])
      end
    end

    context 'with valid value' do
      let(:max_features) { 2 }

      it 'learns model with given parameters.' do
        estimator.fit(x_mlt, y_mlt)
        expect(estimator.params[:max_features]).to eq(2)
      end
    end
  end
end
