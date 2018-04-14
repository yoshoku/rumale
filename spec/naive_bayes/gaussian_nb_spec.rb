# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::NaiveBayes::GaussianNB do
  let(:samples) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:labels) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:estimator) { described_class.new }

  it 'classifies three clusters data.' do
    _n_samples, n_features = samples.shape
    estimator.fit(samples, labels)
    expect(estimator.class_priors.class).to eq(Numo::DFloat)
    expect(estimator.class_priors.shape[0]).to eq(3)
    expect(estimator.class_priors.shape[1]).to be_nil
    expect(estimator.means.class).to eq(Numo::DFloat)
    expect(estimator.means.shape[0]).to eq(3)
    expect(estimator.means.shape[1]).to eq(n_features)
    expect(estimator.variances.class).to eq(Numo::DFloat)
    expect(estimator.variances.shape[0]).to eq(3)
    expect(estimator.variances.shape[1]).to eq(n_features)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(3)
    expect(estimator.score(samples, labels)).to eq(1.0)
  end

  it 'estimates class probabilities with three clusters dataset.' do
    n_samples, _n_features = samples.shape
    estimator.fit(samples, labels)
    probs = estimator.predict_proba(samples)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(3)
    classes = labels.to_a.uniq.sort
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(predicted).to eq(labels)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(samples, labels)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.class_priors).to eq(copied.class_priors)
    expect(estimator.means).to eq(copied.means)
    expect(estimator.variances).to eq(copied.variances)
    expect(copied.score(samples, labels)).to eq(1.0)
  end
end
