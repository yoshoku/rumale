# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::LinearModel::SVR do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:y_mult) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:estimator) { described_class.new(reg_param: 0.01, epsilon: 0.1, random_seed: 1) }
  let(:estimator_bias) { described_class.new(reg_param: 0.01, epsilon: 0.1, fit_bias: true, random_seed: 1) }

  it 'learns the linear model.' do
    n_samples, n_features = x.shape

    estimator.fit(x, y)
    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.size).to eq(n_features)
    expect(estimator.weight_vec.shape[0]).to eq(n_features)
    expect(estimator.weight_vec.shape[1]).to be_nil
    expect(estimator.bias_term).to be_zero

    predicted = estimator.predict(x)
    expect(predicted.class).to eq(Numo::DFloat)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
  end

  it 'learns the linear model with bias term.' do
    _n_samples, n_features = x.shape
    estimator_bias.fit(x, y)
    expect(estimator_bias.weight_vec.size).to eq(n_features)
    expect(estimator_bias.weight_vec.shape[0]).to eq(n_features)
    expect(estimator_bias.weight_vec.shape[1]).to be_nil
    expect(estimator_bias.bias_term).to_not be_zero
    expect(estimator_bias.score(x, y)).to be_within(0.01).of(1.0)
  end

  it 'learns the model for multiple-regression problems.' do
    n_samples, n_features = x.shape
    n_outputs = y_mult.shape[1]

    estimator.fit(x, y_mult)
    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.size).to eq(n_features * n_outputs)
    expect(estimator.weight_vec.shape[0]).to eq(n_features)
    expect(estimator.weight_vec.shape[1]).to eq(n_outputs)

    predicted = estimator.predict(x)
    expect(predicted.class).to eq(Numo::DFloat)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to eq(n_outputs)
    expect(estimator.score(x, y_mult)).to be_within(0.01).of(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x, y)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
    expect(estimator.params[:fit_bias]).to eq(copied.params[:fit_bias])
    expect(estimator.params[:bias_scale]).to eq(copied.params[:bias_scale])
    expect(estimator.params[:epsilon]).to eq(copied.params[:epsilon])
    expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
    expect(estimator.params[:optimizer]).to eq(copied.params[:optimizer])
    expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(estimator.weight_vec).to eq(copied.weight_vec)
    expect(estimator.bias_term).to eq(copied.bias_term)
    expect(estimator.rng).to eq(copied.rng)
    expect(estimator.score(x, y)).to eq(copied.score(x, y))
  end
end
