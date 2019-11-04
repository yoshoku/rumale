# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KNeighborsRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:y) { x[true, 0] + x[true, 1]**2 }
  let(:y_mult) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:estimator) { described_class.new(n_neighbors: 5) }

  it 'learns the model for single regression problem.' do
    n_samples, n_features = x.shape

    estimator.fit(x, y)
    expect(estimator.prototypes.class).to eq(Numo::DFloat)
    expect(estimator.prototypes.shape[0]).to eq(n_samples)
    expect(estimator.prototypes.shape[1]).to eq(n_features)

    expect(estimator.values.class).to eq(Numo::DFloat)
    expect(estimator.values.size).to eq(n_samples)
    expect(estimator.values.shape[0]).to eq(n_samples)
    expect(estimator.values.shape[1]).to be_nil

    predicted = estimator.predict(x)
    expect(predicted.class).to eq(Numo::DFloat)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
  end

  it 'learns the model for multiple regression problem.' do
    n_samples, n_features = x.shape
    n_outputs = y_mult.shape[1]

    estimator.fit(x, y_mult)
    expect(estimator.prototypes.class).to eq(Numo::DFloat)
    expect(estimator.prototypes.shape[0]).to eq(n_samples)
    expect(estimator.prototypes.shape[1]).to eq(n_features)

    expect(estimator.values.class).to eq(Numo::DFloat)
    expect(estimator.values.size).to eq(n_samples * n_outputs)
    expect(estimator.values.shape[0]).to eq(n_samples)
    expect(estimator.values.shape[1]).to eq(n_outputs)

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
    expect(estimator.params[:n_neighbors]).to eq(copied.params[:n_neighbors])
    expect(estimator.prototypes).to eq(copied.prototypes)
    expect(estimator.values).to eq(copied.values)
    expect(copied.score(x, y)).to be_within(0.01).of(1.0)
  end
end
