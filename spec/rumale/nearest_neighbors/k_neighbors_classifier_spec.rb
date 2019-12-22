# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KNeighborsClassifier do
  let(:dataset) { three_clusters_dataset }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { y.to_a.uniq.size }
  let(:estimator) { described_class.new(n_neighbors: 5, algorithm: algorithm, metric: metric).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when metric is "euclidean"' do
    let(:metric) { 'euclidean' }
    let(:x) { dataset[0] }

    context 'when algorithm is "brute"' do
      let(:algorithm) { 'brute' }

      it 'classifies three clusters data.', :aggregate_failures do
        expect(estimator.prototypes.class).to eq(Numo::DFloat)
        expect(estimator.prototypes.ndim).to eq(2)
        expect(estimator.prototypes.shape[0]).to eq(n_samples)
        expect(estimator.prototypes.shape[1]).to eq(n_features)
        expect(estimator.labels.class).to eq(Numo::Int32)
        expect(estimator.labels.ndim).to eq(1)
        expect(estimator.labels.shape[0]).to eq(n_samples)
        expect(estimator.classes.class).to eq(Numo::Int32)
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(predicted.class).to eq(Numo::Int32)
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted).to eq(y)
        expect(score).to eq(1.0)
      end

      it 'dumps and restores itself using Marshal module.', :aggregate_failures do
        expect(estimator.class).to eq(copied.class)
        expect(estimator.params).to eq(copied.params)
        expect(estimator.prototypes).to eq(copied.prototypes)
        expect(estimator.labels).to eq(copied.labels)
        expect(estimator.classes).to eq(copied.classes)
        expect(score).to eq(copied.score(x, y))
      end
    end

    context 'when algorithm is "vptree"' do
      let(:algorithm) { 'vptree' }

      it 'classifies three clusters data.', :aggregate_failures do
        expect(estimator.prototypes.class).to eq(Rumale::NearestNeighbors::VPTree)
        expect(estimator.labels.class).to eq(Numo::Int32)
        expect(estimator.labels.ndim).to eq(1)
        expect(estimator.labels.shape[0]).to eq(n_samples)
        expect(estimator.classes.class).to eq(Numo::Int32)
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(predicted.class).to eq(Numo::Int32)
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(score).to be_within(0.05).of(1.0)
      end

      it 'dumps and restores itself using Marshal module.', :aggregate_failures do
        expect(estimator.class).to eq(copied.class)
        expect(estimator.params).to eq(copied.params)
        expect(estimator.prototypes.class).to eq(copied.prototypes.class)
        expect(estimator.labels).to eq(copied.labels)
        expect(estimator.classes).to eq(copied.classes)
        expect(score).to eq(copied.score(x, y))
      end
    end
  end

  context 'when metric is "precomputed"' do
    let(:algorithm) { 'vptree' }
    let(:metric) { 'precomputed' }
    let(:x) { Rumale::PairwiseMetric.manhattan_distance(dataset[0]) }

    it 'classifies three clusters data.', :aggregate_failures do
      expect(estimator.prototypes.class).to eq(NilClass)
      expect(estimator.labels.class).to eq(Numo::Int32)
      expect(estimator.labels.ndim).to eq(1)
      expect(estimator.labels.shape[0]).to eq(n_samples)
      expect(estimator.classes.class).to eq(Numo::Int32)
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(score).to eq(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params[:n_neighbors]).to eq(copied.params[:n_neighbors])
      expect(estimator.prototypes).to eq(copied.prototypes)
      expect(estimator.labels).to eq(copied.labels)
      expect(estimator.classes).to eq(copied.classes)
      expect(score).to eq(copied.score(x, y))
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
