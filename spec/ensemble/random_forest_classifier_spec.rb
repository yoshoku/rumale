# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::Ensemble::RandomForestClassifier do
  let(:samples) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:labels) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:n_estimators) { 10 }
  let(:estimator) { described_class.new(n_estimators: n_estimators, random_seed: 1) }

  it 'classifies three clusters data.' do
    _n_samples, n_features = samples.shape
    estimator.fit(samples, labels)
    expect(estimator.estimators.class).to eq(Array)
    expect(estimator.estimators.size).to eq(n_estimators)
    expect(estimator.estimators[0].class).to eq(SVMKit::Tree::DecisionTreeClassifier)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(3)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(samples, labels)).to eq(1.0)
  end

  it 'estimates class probabilities with three clusters dataset.' do
    n_samples, = samples.shape
    estimator.fit(samples, labels)
    probs = estimator.predict_proba(samples)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(3)
    classes = labels.to_a.uniq.sort
    predicted = Numo::Int32.asarray(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })
    expect(predicted).to eq(labels)
  end

  it 'returns leaf index that each sample reached' do
    n_samples, = samples.shape
    estimator.fit(samples, labels)
    index_mat = estimator.apply(samples)
    expect(index_mat.shape[0]).to eq(n_samples)
    expect(index_mat.shape[1]).to eq(n_estimators)
    expect(index_mat[true, 0]).to eq(estimator.estimators[0].apply(samples))
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(samples, labels)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.estimators.size).to eq(copied.estimators.size)
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.feature_importances).to eq(copied.feature_importances)
    expect(estimator.rng).to eq(copied.rng)
    expect(copied.score(samples, labels)).to eq(1.0)
  end
end
