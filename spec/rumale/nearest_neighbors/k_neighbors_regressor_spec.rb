# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KNeighborsRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:single_target) { x[true, 0] + x[true, 1]**2 }
  let(:multi_target) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { y.shape[1] }
  let(:estimator) { described_class.new(n_neighbors: 5).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when single target problem' do
    let(:y) { single_target }

    it 'learns the model for single regression problem.', :aggregate_failures do
      expect(estimator.prototypes.class).to eq(Numo::DFloat)
      expect(estimator.prototypes.ndim).to eq(2)
      expect(estimator.prototypes.shape[0]).to eq(n_samples)
      expect(estimator.prototypes.shape[1]).to eq(n_features)
      expect(estimator.values.class).to eq(Numo::DFloat)
      expect(estimator.values.ndim).to eq(1)
      expect(estimator.values.shape[0]).to eq(n_samples)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params[:n_neighbors]).to eq(copied.params[:n_neighbors])
      expect(estimator.prototypes).to eq(copied.prototypes)
      expect(estimator.values).to eq(copied.values)
      expect(score).to eq(copied.score(x, y))
    end
  end

  context 'when multi-target problem' do
    let(:y) { multi_target }

    it 'learns the model for multiple regression problem.', :aggregate_failures do
      expect(estimator.prototypes.class).to eq(Numo::DFloat)
      expect(estimator.prototypes.ndim).to eq(2)
      expect(estimator.prototypes.shape[0]).to eq(n_samples)
      expect(estimator.prototypes.shape[1]).to eq(n_features)
      expect(estimator.values.class).to eq(Numo::DFloat)
      expect(estimator.values.ndim).to eq(2)
      expect(estimator.values.shape[0]).to eq(n_samples)
      expect(estimator.values.shape[1]).to eq(n_outputs)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.01).of(1.0)
    end
  end
end
