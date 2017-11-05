require 'spec_helper'

RSpec.describe SVMKit::LinearModel::LogisticRegression do
  let(:samples) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:labels) { Marshal.load(File.read(__dir__ + '/../test_labels.dat')) }
  let(:estimator) { described_class.new(reg_param: 1.0, max_iter: 100, batch_size: 20, random_seed: 1) }
  let(:estimator_bias) do
    described_class.new(reg_param: 1.0, fit_bias: true, max_iter: 100, batch_size: 20, random_seed: 1)
  end

  it 'classifies two clusters.' do
    n_samples, n_features = samples.shape
    estimator.fit(samples, labels)
    expect(estimator.weight_vec.size).to eq(n_features)
    expect(estimator.weight_vec.shape[0]).to eq(n_features)
    expect(estimator.weight_vec.shape[1]).to be_nil
    expect(estimator.bias_term).to be_zero

    func_vals = estimator.decision_function(samples)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to be_nil

    predicted = estimator.predict(samples)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(predicted).to eq(labels)

    score = estimator.score(samples, labels)
    expect(score).to eq(1.0)
  end

  it 'learns the model of two clusters dataset with bias term.' do
    _n_samples, n_features = samples.shape
    estimator_bias.fit(samples, labels)
    expect(estimator_bias.weight_vec.size).to eq(n_features)
    expect(estimator_bias.weight_vec.shape[0]).to eq(n_features)
    expect(estimator_bias.weight_vec.shape[1]).to be_nil
    expect(estimator_bias.bias_term).to_not be_zero
    score = estimator_bias.score(samples, labels)
    expect(score).to eq(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(samples, labels)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
    expect(estimator.params[:fit_bias]).to eq(copied.params[:fit_bias])
    expect(estimator.params[:bias_scale]).to eq(copied.params[:bias_scale])
    expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
    expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(estimator.weight_vec).to eq(copied.weight_vec)
    expect(estimator.bias_term).to eq(copied.bias_term)
    expect(estimator.rng).to eq(copied.rng)
  end
end
