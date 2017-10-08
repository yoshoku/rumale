require 'spec_helper'

RSpec.describe SVMKit::LinearModel::LogisticRegression do
  let(:samples) { SVMKit::Utils.restore_nmatrix(Marshal.load(File.read(__dir__ + '/test_samples.dat'))) }
  let(:labels) { SVMKit::Utils.restore_nmatrix(Marshal.load(File.read(__dir__ + '/test_labels.dat'))) }
  let(:estimator) { described_class.new(penalty: 1.0, max_iter: 100, batch_size: 20, random_seed: 1) }
  let(:estimator_bias) {
    described_class.new(penalty: 1.0, fit_bias: true, max_iter: 100, batch_size: 20, random_seed: 1) }

  it 'classifies two clusters.' do
    n_samples, n_features = samples.shape
    estimator.fit(samples, labels)
    expect(estimator.weight_vec.size).to eq(n_features)
    expect(estimator.bias_term).to eq(0.0)
    score = estimator.score(samples, labels)
    expect(score).to eq(1.0)
  end

  it 'learns the model of two clusters dataset with bias term.' do
    n_samples, n_features = samples.shape
    estimator_bias.fit(samples, labels)
    expect(estimator_bias.weight_vec.size).to eq(n_features)
    expect(estimator_bias.bias_term).to_not eq(0.0)
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
