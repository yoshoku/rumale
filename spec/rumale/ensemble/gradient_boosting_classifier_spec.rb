# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::GradientBoostingClassifier do
  let(:x_bin) { Marshal.load(File.read(__dir__ + '/../../test_samples.dat')) }
  let(:y_bin) { Marshal.load(File.read(__dir__ + '/../../test_labels.dat')) }
  let(:x_mlt) { Marshal.load(File.read(__dir__ + '/../../test_samples_three_clusters.dat')) }
  let(:y_mlt) { Marshal.load(File.read(__dir__ + '/../../test_labels_three_clusters.dat')) }
  let(:n_estimators) { 10 }
  let(:estimator) { described_class.new(n_estimators: n_estimators, learning_rate: 0.9, max_features: 1, random_seed: 1) }
  let(:estimator_parallel) do
    described_class.new(n_estimators: n_estimators, learning_rate: 0.9, max_features: 1, n_jobs: -1, random_seed: 1)
  end

  it 'classifies two clusters data.' do
    n_samples, n_features = x_bin.shape
    estimator.fit(x_bin, y_bin)
    leaf_ids = estimator.apply(x_bin)
    expect(estimator.estimators.class).to eq(Array)
    expect(estimator.estimators[0].class).to eq(Rumale::Tree::GradientTreeRegressor)
    expect(estimator.estimators.size).to eq(n_estimators)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(2)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(x_bin, y_bin)).to be_within(0.02).of(1.0)
    expect(leaf_ids.class).to eq(Numo::Int32)
    expect(leaf_ids.shape).to eq([n_samples, n_estimators])
  end

  it 'classifies three clusters data.' do
    n_samples, n_features = x_mlt.shape
    estimator.fit(x_mlt, y_mlt)
    leaf_ids = estimator.apply(x_mlt)
    expect(estimator.estimators.class).to eq(Array)
    expect(estimator.estimators[0].class).to eq(Array)
    expect(estimator.estimators[0][0].class).to eq(Rumale::Tree::GradientTreeRegressor)
    expect(estimator.estimators.size).to eq(3)
    expect(estimator.estimators[0].size).to eq(n_estimators)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(3)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(x_mlt, y_mlt)).to be_within(0.02).of(1.0)
    expect(leaf_ids.class).to eq(Numo::Int32)
    expect(leaf_ids.shape).to eq([n_samples, n_estimators, 3])
  end

  it 'classifies three clusters data in parallel.' do
    n_samples, n_features = x_mlt.shape
    estimator_parallel.fit(x_mlt, y_mlt)
    leaf_ids = estimator_parallel.apply(x_mlt)
    expect(estimator_parallel.estimators.class).to eq(Array)
    expect(estimator_parallel.estimators[0].class).to eq(Array)
    expect(estimator_parallel.estimators[0][0].class).to eq(Rumale::Tree::GradientTreeRegressor)
    expect(estimator_parallel.estimators.size).to eq(3)
    expect(estimator_parallel.estimators[0].size).to eq(n_estimators)
    expect(estimator_parallel.classes.class).to eq(Numo::Int32)
    expect(estimator_parallel.classes.size).to eq(3)
    expect(estimator_parallel.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator_parallel.feature_importances.shape[0]).to eq(n_features)
    expect(estimator_parallel.feature_importances.shape[1]).to be_nil
    expect(estimator_parallel.score(x_mlt, y_mlt)).to be_within(0.02).of(1.0)
    expect(leaf_ids.class).to eq(Numo::Int32)
    expect(leaf_ids.shape).to eq([n_samples, n_estimators, 3])
  end

  it 'estimates class probabilities with three clusters dataset.' do
    n_samples = x_mlt.shape[0]
    estimator.fit(x_mlt, y_mlt)
    probs = estimator.predict_proba(x_mlt)
    classes = y_mlt.to_a.uniq.sort
    predicted = Numo::Int32.asarray(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(3)
    expect(predicted).to eq(y_mlt)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x_mlt, y_mlt)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params).to match(copied.params)
    expect(estimator.estimators.size).to eq(copied.estimators.size)
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.feature_importances).to eq(copied.feature_importances)
    expect(estimator.rng).to eq(copied.rng)
    expect(estimator.score(x_mlt, y_mlt)).to eq(copied.score(x_mlt, y_mlt))
  end
end
