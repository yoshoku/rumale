# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Optimizer::SGD do
  let(:x) { two_clusters_dataset[0] }
  let(:y) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:y_mult) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:optimizer) { described_class.new(learning_rate: 0.1, momentum: 0.9, decay: 0.1) }
  let(:estimator) { Rumale::LinearModel::LinearRegression.new(optimizer: optimizer, max_iter: 100, random_seed: 1) }

  it 'learns the model for single regression problem with optimizer.' do
    estimator.fit(x, y)
    expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
  end

  it 'learns the model for multiple-regression problems with optimizer.' do
    estimator.fit(x, y_mult)
    expect(estimator.score(x, y_mult)).to be_within(0.01).of(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    optimizer.call(Numo::DFloat.new(3).rand, Numo::DFloat.new(3).rand)
    copied = Marshal.load(Marshal.dump(optimizer))
    expect(optimizer.class).to eq(copied.class)
    expect(optimizer.params).to eq(copied.params)
    expect(optimizer.instance_variable_get(:@update)).to eq(copied.instance_variable_get(:@update))
    expect(optimizer.instance_variable_get(:@iter)).to eq(copied.instance_variable_get(:@iter))
  end
end
