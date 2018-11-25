# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::Ensemble::AdaBoostClassifier do
  let(:x_bin) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y_bin) { Marshal.load(File.read(__dir__ + '/../test_labels.dat')) }
  let(:x_mlt) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:y_mlt) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:estimator) { described_class.new(n_estimators: 10, max_features: 1, random_seed: 1) }

  it 'classifies two clusters data.' do
    _n_samples, n_features = x_bin.shape
    estimator.fit(x_bin, y_bin)
    expect(estimator.estimators.class).to eq(Array)
    expect(estimator.estimators[0].class).to eq(SVMKit::Tree::DecisionTreeClassifier)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(2)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(x_bin, y_bin)).to be_within(0.021).of(1.0)
  end

  it 'classifies three clusters data.' do
    _n_samples, n_features = x_mlt.shape
    estimator.fit(x_mlt, y_mlt)
    expect(estimator.estimators.class).to eq(Array)
    expect(estimator.estimators[0].class).to eq(SVMKit::Tree::DecisionTreeClassifier)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(3)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.score(x_mlt, y_mlt)).to be_within(0.02).of(1.0)
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

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x_mlt, y_mlt)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.estimators.size).to eq(copied.estimators.size)
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.feature_importances).to eq(copied.feature_importances)
    expect(estimator.rng).to eq(copied.rng)
    expect(copied.score(x_mlt, y_mlt)).to be_within(0.02).of(1.0)
  end
end
