# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NaiveBayes::GaussianNB do
  let(:three_clusters) { three_clusters_dataset }
  let(:x) { three_clusters[0] }
  let(:y) { three_clusters[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_classes) { classes.size }
  let(:estimator) { described_class.new.fit(x, y) }
  let(:probs) { estimator.predict_proba(x) }
  let(:score) { estimator.score(x, y) }
  let(:func_vals) { estimator.decision_function(x) }
  let(:predicted) { estimator.predict(x) }
  let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  it 'classifies three clusters data.', :aggregate_failures do
    expect(estimator.class_priors.class).to eq(Numo::DFloat)
    expect(estimator.class_priors.ndim).to eq(1)
    expect(estimator.class_priors.shape[0]).to eq(n_classes)
    expect(estimator.means.class).to eq(Numo::DFloat)
    expect(estimator.means.ndim).to eq(2)
    expect(estimator.means.shape[0]).to eq(n_classes)
    expect(estimator.means.shape[1]).to eq(n_features)
    expect(estimator.variances.class).to eq(Numo::DFloat)
    expect(estimator.variances.ndim).to eq(2)
    expect(estimator.variances.shape[0]).to eq(n_classes)
    expect(estimator.variances.shape[1]).to eq(n_features)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.ndim).to eq(1)
    expect(estimator.classes.shape[0]).to eq(n_classes)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.ndim).to eq(2)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to eq(n_classes)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.ndim).to eq(1)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted).to eq(y)
    expect(score).to eq(1.0)
  end

  it 'estimates class probabilities with three clusters dataset.', :aggregate_failures do
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.ndim).to eq(2)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(n_classes)
    expect(predicted_by_probs).to eq(y)
  end

  it 'dumps and restores itself using Marshal module.', :aggregate_failures do
    expect(estimator.class).to eq(copied.class)
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.class_priors).to eq(copied.class_priors)
    expect(estimator.means).to eq(copied.means)
    expect(estimator.variances).to eq(copied.variances)
    expect(score).to eq(copied.score(x, y))
  end
end
