# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Multiclass::OneVsRestClassifier do
  let(:three_clusters) { three_clusters_dataset }
  let(:samples) { three_clusters[0] }
  let(:labels) { three_clusters[1] }
  let(:base_estimator) { Rumale::LinearModel::SVC.new(reg_param: 1.0, max_iter: 100, batch_size: 20, random_seed: 1) }
  let(:estimator) { described_class.new(estimator: base_estimator) }

  it 'classifies three clusters.' do
    n_classes = labels.to_a.uniq.size
    n_samples, = samples.shape
    estimator.fit(samples, labels)

    expect(estimator.estimators.size).to eq(n_classes)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.shape[0]).to eq(n_classes)
    expect(estimator.classes.shape[1]).to be_nil

    func_vals = estimator.decision_function(samples)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to eq(n_classes)

    predicted = estimator.predict(samples)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(predicted).to eq(labels)

    expect(estimator.score(samples, labels)).to eq(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(samples, labels)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.estimators.size).to eq(copied.estimators.size)
    expect(estimator.estimators[0].class).to eq(copied.estimators[0].class)
    expect(estimator.estimators[1].class).to eq(copied.estimators[1].class)
    expect(estimator.estimators[2].class).to eq(copied.estimators[2].class)
    expect(estimator.estimators[0].weight_vec).to eq(copied.estimators[0].weight_vec)
    expect(estimator.estimators[1].weight_vec).to eq(copied.estimators[1].weight_vec)
    expect(estimator.estimators[2].weight_vec).to eq(copied.estimators[2].weight_vec)
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.params[:estimator].class).to eq(copied.params[:estimator].class)
    expect(copied.score(samples, labels)).to eq(1.0)
  end
end
