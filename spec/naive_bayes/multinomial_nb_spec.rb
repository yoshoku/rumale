require 'spec_helper'

RSpec.describe SVMKit::NaiveBayes::MultinomialNB do
  let(:samples) { Numo::DFloat[[4, 3, 0, 0], [4, 0, 0, 0], [4, 0, 1, 0], [0, 0, 5, 3], [0, 0, 0, 3], [0, 1, 5, 3]] }
  let(:labels) { Numo::Int32[1, 1, 1, -1, -1, -1] }
  let(:estimator) { described_class.new(smoothing_param: 1.0) }

  it 'classifies two clusters data.' do
    _n_samples, n_features = samples.shape
    estimator.fit(samples, labels)
    expect(estimator.class_priors.class).to eq(Numo::DFloat)
    expect(estimator.class_priors.shape[0]).to eq(2)
    expect(estimator.class_priors.shape[1]).to be_nil
    expect(estimator.feature_probs.class).to eq(Numo::DFloat)
    expect(estimator.feature_probs.shape[0]).to eq(2)
    expect(estimator.feature_probs.shape[1]).to eq(n_features)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(2)
    expect(estimator.score(samples, labels)).to eq(1.0)
  end

  it 'estimates class probabilities with two clusters dataset.' do
    n_samples, _n_features = samples.shape
    estimator.fit(samples, labels)
    probs = estimator.predict_proba(samples)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(2)
    classes = labels.to_a.uniq.sort
    predicted = Numo::Int32[*Array.new(n_samples) { |n| classes[probs[n, true].max_index] }]
    expect(predicted).to eq(labels)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(samples, labels)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params[:smoothing_param]).to eq(copied.params[:smoothing_param])
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.class_priors).to eq(copied.class_priors)
    expect(estimator.feature_probs).to eq(copied.feature_probs)
  end
end
