# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KNeighborsRegressor do
  let(:dataset) { two_clusters_dataset }
  let(:single_target) { dataset[0][true, 0] + dataset[0][true, 1]**2 }
  let(:multi_target) { dataset[0].dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_outputs) { y.shape[1] }
  let(:estimator) { described_class.new(n_neighbors: 5, metric: metric).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  context 'when metric is "euclidean"' do
    let(:metric) { 'euclidean' }
    let(:x) { dataset[0] }
    let(:n_features) { x.shape[1] }

    context 'when single target problem' do
      let(:y) { single_target }

      it 'learns the model for single regression problem', :aggregate_failures do
        expect(estimator.prototypes).to be_a(Numo::DFloat)
        expect(estimator.prototypes).to be_contiguous
        expect(estimator.prototypes.ndim).to eq(2)
        expect(estimator.prototypes.shape[0]).to eq(n_samples)
        expect(estimator.prototypes.shape[1]).to eq(n_features)
        expect(estimator.values).to be_a(Numo::DFloat)
        expect(estimator.values).to be_contiguous
        expect(estimator.values.ndim).to eq(1)
        expect(estimator.values.shape[0]).to eq(n_samples)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(score).to be_within(0.01).of(1.0)
      end
    end

    context 'when multi-target problem' do
      let(:y) { multi_target }

      it 'learns the model for multiple regression problem', :aggregate_failures do
        expect(estimator.prototypes).to be_a(Numo::DFloat)
        expect(estimator.prototypes).to be_contiguous
        expect(estimator.prototypes.ndim).to eq(2)
        expect(estimator.prototypes.shape[0]).to eq(n_samples)
        expect(estimator.prototypes.shape[1]).to eq(n_features)
        expect(estimator.values).to be_a(Numo::DFloat)
        expect(estimator.values).to be_contiguous
        expect(estimator.values.ndim).to eq(2)
        expect(estimator.values.shape[0]).to eq(n_samples)
        expect(estimator.values.shape[1]).to eq(n_outputs)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.01).of(1.0)
      end
    end
  end

  context 'when metric is "precomputed"' do
    let(:metric) { 'precomputed' }
    let(:x) { Rumale::PairwiseMetric.manhattan_distance(dataset[0]) }

    context 'when single target problem' do
      let(:y) { single_target }

      it 'learns the model for single regression problem', :aggregate_failures do
        expect(estimator.prototypes).to be_nil
        expect(estimator.values).to be_a(Numo::DFloat)
        expect(estimator.values).to be_contiguous
        expect(estimator.values.ndim).to eq(1)
        expect(estimator.values.shape[0]).to eq(n_samples)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(score).to be_within(0.01).of(1.0)
      end
    end

    context 'when multi-target problem' do
      let(:y) { multi_target }

      it 'learns the model for multiple regression problem', :aggregate_failures do
        expect(estimator.prototypes).to be_nil
        expect(estimator.values).to be_a(Numo::DFloat)
        expect(estimator.values).to be_contiguous
        expect(estimator.values.ndim).to eq(2)
        expect(estimator.values.shape[0]).to eq(n_samples)
        expect(estimator.values.shape[1]).to eq(n_outputs)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.01).of(1.0)
      end
    end

    context 'when wrong size matrix is given' do
      let(:estimator) { described_class.new(n_neighbors: 5, metric: 'precomputed') }
      let(:y) { single_target }

      it 'raises ArgumentError', :aggregate_failures do
        expect { estimator.fit(Numo::DFloat.new(n_samples, 2).rand, y) }.to raise_error(ArgumentError)
        expect { estimator.fit(x, y).predict(Numo::DFloat.new(2, n_samples / 2).rand) }.to raise_error(ArgumentError)
      end
    end
  end
end
