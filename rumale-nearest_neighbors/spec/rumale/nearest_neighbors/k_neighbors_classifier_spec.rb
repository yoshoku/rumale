# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KNeighborsClassifier do
  let(:dataset) { three_clusters_dataset }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { y.to_a.uniq.size }
  let(:estimator) { described_class.new(n_neighbors: 5, metric: metric).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  context 'when metric is "euclidean"' do
    let(:metric) { 'euclidean' }
    let(:x) { dataset[0] }

    it 'classifies three clusters data', :aggregate_failures do
      expect(estimator.prototypes).to be_a(Numo::DFloat)
      expect(estimator.prototypes).to be_contiguous
      expect(estimator.prototypes.ndim).to eq(2)
      expect(estimator.prototypes.shape[0]).to eq(n_samples)
      expect(estimator.prototypes.shape[1]).to eq(n_features)
      expect(estimator.labels).to be_a(Numo::Int32)
      expect(estimator.labels).to be_contiguous
      expect(estimator.labels.ndim).to eq(1)
      expect(estimator.labels.shape[0]).to eq(n_samples)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(score).to eq(1.0)
    end
  end

  context 'when metric is "precomputed"' do
    let(:metric) { 'precomputed' }
    let(:x) { Rumale::PairwiseMetric.manhattan_distance(dataset[0]) }

    it 'classifies three clusters data', :aggregate_failures do
      expect(estimator.prototypes).to be_nil
      expect(estimator.labels).to be_a(Numo::Int32)
      expect(estimator.labels).to be_contiguous
      expect(estimator.labels.ndim).to eq(1)
      expect(estimator.labels.shape[0]).to eq(n_samples)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(score).to eq(1.0)
    end

    context 'when wrong size matrix is given' do
      let(:estimator) { described_class.new(n_neighbors: 5, metric: 'precomputed') }

      it 'raises ArgumentError', :aggregate_failures do
        expect { estimator.fit(Numo::DFloat.new(n_samples, 2).rand, y) }.to raise_error(ArgumentError)
        expect { estimator.fit(x, y).predict(Numo::DFloat.new(2, n_samples / 2).rand) }.to raise_error(ArgumentError)
        expect { estimator.fit(x, y).decision_function(Numo::DFloat.new(2, n_samples / 2).rand) }.to raise_error(ArgumentError)
      end
    end
  end
end
