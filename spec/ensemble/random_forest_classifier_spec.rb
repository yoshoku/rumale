# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::RandomForestClassifier do
  let(:x_bin) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y_bin) { Marshal.load(File.read(__dir__ + '/../test_labels.dat')) }
  let(:x_mlt) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:y_mlt) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:n_estimators) { 10 }
  let(:estimator) { described_class.new(n_estimators: n_estimators, max_depth: 2, max_features: 2, random_seed: 1) }
  let(:estimator_parallel) do
    described_class.new(n_estimators: n_estimators, max_depth: 2, max_features: 2, n_jobs: -1, random_seed: 1)
  end

  it 'classifies two clusters data.' do
    _n_samples, n_features = x_bin.shape
    estimator.fit(x_bin, y_bin)
    expect(estimator.params[:n_estimators]).to eq(n_estimators)
    expect(estimator.params[:max_depth]).to eq(2)
    expect(estimator.params[:max_features]).to eq(2)
    expect(estimator.estimators.class).to eq(Array)
    expect(estimator.estimators.size).to eq(n_estimators)
    expect(estimator.estimators[0].class).to eq(Rumale::Tree::DecisionTreeClassifier)
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
    expect(estimator.estimators.class).to eq(Array)
    expect(estimator.estimators.size).to eq(n_estimators)
    expect(estimator.estimators[0].class).to eq(Rumale::Tree::DecisionTreeClassifier)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(3)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(x_mlt, y_mlt)).to eq(1.0)
  end

  it 'estimates class probabilities with three clusters dataset.' do
    n_samples, = x_mlt.shape
    estimator.fit(x_mlt, y_mlt)
    probs = estimator.predict_proba(x_mlt)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(3)
    classes = y_mlt.to_a.uniq.sort
    predicted = Numo::Int32.asarray(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })
    expect(predicted).to eq(y_mlt)
  end

  it 'classifies three clusters data in parallel.' do
    _n_samples, n_features = x_mlt.shape
    estimator_parallel.fit(x_mlt, y_mlt)
    expect(estimator_parallel.estimators.class).to eq(Array)
    expect(estimator_parallel.estimators.size).to eq(n_estimators)
    expect(estimator_parallel.estimators[0].class).to eq(Rumale::Tree::DecisionTreeClassifier)
    expect(estimator_parallel.classes.class).to eq(Numo::Int32)
    expect(estimator_parallel.classes.size).to eq(3)
    expect(estimator_parallel.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator_parallel.feature_importances.shape[0]).to eq(n_features)
    expect(estimator_parallel.feature_importances.shape[1]).to be_nil
    expect(estimator_parallel.score(x_mlt, y_mlt)).to eq(1.0)
  end

  it 'returns leaf index that each sample reached' do
    n_samples, = x_mlt.shape
    estimator.fit(x_mlt, y_mlt)
    index_mat = estimator.apply(x_mlt)
    expect(index_mat.shape[0]).to eq(n_samples)
    expect(index_mat.shape[1]).to eq(n_estimators)
    expect(index_mat[true, 0]).to eq(estimator.estimators[0].apply(x_mlt))
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
    expect(copied.score(x_mlt, y_mlt)).to eq(1.0)
  end
end
