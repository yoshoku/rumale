# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::PolynomialModel::FactorizationMachineRegressor do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y) { x.dot(Numo::DFloat[0.8, 0.2]) }
  let(:y_mult) { x.dot(Numo::DFloat[[0.8, 0.82], [0.2, 0.18]]) }
  let(:n_factors) { 2 }
  let(:estimator) { described_class.new(n_factors: n_factors, reg_param_linear: 0.1, reg_param_factor: 0.1, random_seed: 1) }

  it 'learns the the model for single regression problem.' do
    n_samples, n_features = x.shape

    estimator.fit(x, y)
    expect(estimator.factor_mat.class).to eq(Numo::DFloat)
    expect(estimator.factor_mat.size).to eq(n_factors * n_features)
    expect(estimator.factor_mat.shape[0]).to eq(n_factors)
    expect(estimator.factor_mat.shape[1]).to eq(n_features)
    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.size).to eq(n_features)
    expect(estimator.weight_vec.shape[0]).to eq(n_features)
    expect(estimator.weight_vec.shape[1]).to be_nil
    expect(estimator.bias_term.class).to eq(Float)

    predicted = estimator.predict(x)
    expect(predicted.class).to eq(Numo::DFloat)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
  end

  it 'learns the model for multiple-regression problem.' do
    n_samples, n_features = x.shape
    n_outputs = y_mult.shape[1]

    estimator.fit(x, y_mult)
    expect(estimator.factor_mat.class).to eq(Numo::DFloat)
    expect(estimator.factor_mat.size).to eq(n_outputs * n_factors * n_features)
    expect(estimator.factor_mat.shape[0]).to eq(n_outputs)
    expect(estimator.factor_mat.shape[1]).to eq(n_factors)
    expect(estimator.factor_mat.shape[2]).to eq(n_features)
    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.size).to eq(n_features * n_outputs)
    expect(estimator.weight_vec.shape[0]).to eq(n_features)
    expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
    expect(estimator.bias_term.class).to eq(Numo::DFloat)
    expect(estimator.bias_term.size).to eq(n_outputs)
    expect(estimator.bias_term.shape[0]).to eq(n_outputs)
    expect(estimator.bias_term.shape[1]).to be_nil

    predicted = estimator.predict(x)
    expect(predicted.class).to eq(Numo::DFloat)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to eq(n_outputs)
    expect(estimator.score(x, y_mult)).to be_within(0.01).of(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x, y)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.factor_mat).to eq(copied.factor_mat)
    expect(estimator.weight_vec).to eq(copied.weight_vec)
    expect(estimator.bias_term).to eq(copied.bias_term)
    expect(estimator.rng).to eq(copied.rng)
    expect(estimator.params[:n_factors]).to eq(copied.params[:n_factors])
    expect(estimator.params[:reg_param_linear]).to eq(copied.params[:reg_param_linear])
    expect(estimator.params[:reg_param_factor]).to eq(copied.params[:reg_param_factor])
    expect(estimator.params[:learning_rate]).to eq(copied.params[:learning_rate])
    expect(estimator.params[:decay]).to eq(copied.params[:decay])
    expect(estimator.params[:momentum]).to eq(copied.params[:momentum])
    expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
    expect(estimator.params[:optimizer].class).to eq(copied.params[:optimizer].class)
    expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(estimator.score(x, y)).to eq(copied.score(x, y))
  end
end
