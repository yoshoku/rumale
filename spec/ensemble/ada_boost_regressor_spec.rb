# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::Ensemble::AdaBoostRegressor do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y) { x[true, 0] + x[true, 1]**2 }
  let(:y_mult) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.0, 0.1]]) }
  let(:n_estimators) { 10 }
  let(:estimator) { described_class.new(n_estimators: n_estimators, criterion: 'mae', max_features: 2, random_seed: 9) }

  it 'learns the model for single regression problem.' do
    n_samples, n_features = x.shape

    estimator.fit(x, y)

    expect(estimator.estimators.class).to eq(Array)
    expect(estimator.estimators[0].class).to eq(SVMKit::Tree::DecisionTreeRegressor)
    expect(estimator.feature_importances.class).to eq(Numo::DFloat)
    expect(estimator.feature_importances.shape[0]).to eq(n_features)
    expect(estimator.feature_importances.shape[1]).to be_nil
    expect(estimator.estimator_weights.class).to eq(Numo::DFloat)
    expect(estimator.estimator_weights.shape[0]).to eq(estimator.estimators.size)
    expect(estimator.estimator_weights.shape[1]).to be_nil

    predicted = estimator.predict(x)
    expect(predicted.class).to eq(Numo::DFloat)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
  end

  it 'raises ArgumentError when given multiple target values.' do
    expect { estimator.fit(x, y_mult) }.to raise_error(ArgumentError)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x, y)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.estimators.size).to eq(copied.estimators.size)
    expect(estimator.estimator_weights).to eq(copied.estimator_weights)
    expect(estimator.feature_importances).to eq(copied.feature_importances)
    expect(estimator.rng).to eq(copied.rng)
    expect(estimator.score(x, y)).to eq(copied.score(x, y))
  end
end
