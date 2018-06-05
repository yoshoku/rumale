# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::Optimizer::YellowFin do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:y_mult) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:optimizer) { described_class.new(learning_rate: 0.05, momentum: 0.8, decay: 0.995, window_width: 2) }
  let(:estimator) { SVMKit::LinearModel::LinearRegression.new(optimizer: optimizer, max_iter: 100, random_seed: 1) }

  it 'learns the model for single regression problem with optimizer.' do
    estimator.fit(x, y)
    expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
  end

  it 'learns the model for multiple-regression problems with optimizer.' do
    estimator.fit(x, y_mult)
    expect(estimator.score(x, y_mult)).to be_within(0.01).of(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    10.times { optimizer.call(Numo::DFloat[1, 2, 3], Numo::DFloat[0.1, 0.2, 0.3]) }
    copied = Marshal.load(Marshal.dump(optimizer))
    expect(optimizer.class).to eq(copied.class)
    expect(optimizer.params).to eq(copied.params)
    expect(optimizer.instance_variable_get(:@smth_learning_rate)).to eq(copied.instance_variable_get(:@smth_learning_rate))
    expect(optimizer.instance_variable_get(:@smth_momentum)).to eq(copied.instance_variable_get(:@smth_momentum))
    expect(optimizer.instance_variable_get(:@grad_norms)).to eq(copied.instance_variable_get(:@grad_norms))
    expect(optimizer.instance_variable_get(:@grad_norm_min)).to eq(copied.instance_variable_get(:@grad_norm_min))
    expect(optimizer.instance_variable_get(:@grad_norm_min)).to eq(copied.instance_variable_get(:@grad_norm_min))
    expect(optimizer.instance_variable_get(:@grad_norm_max)).to eq(copied.instance_variable_get(:@grad_norm_max))
    expect(optimizer.instance_variable_get(:@grad_mean_sqr)).to eq(copied.instance_variable_get(:@grad_mean_sqr))
    expect(optimizer.instance_variable_get(:@grad_mean)).to eq(copied.instance_variable_get(:@grad_mean))
    expect(optimizer.instance_variable_get(:@grad_var)).to eq(copied.instance_variable_get(:@grad_var))
    expect(optimizer.instance_variable_get(:@grad_norm_mean)).to eq(copied.instance_variable_get(:@grad_norm_mean))
    expect(optimizer.instance_variable_get(:@curve_mean)).to eq(copied.instance_variable_get(:@curve_mean))
    expect(optimizer.instance_variable_get(:@distance_mean)).to eq(copied.instance_variable_get(:@distance_mean))
    expect(optimizer.instance_variable_get(:@update)).to eq(copied.instance_variable_get(:@update))
  end
end
