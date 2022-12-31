# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::MetricLearning::MLKR do
  let(:dataset) { regression_dataset }
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_components) { nil }
  let(:init) { 'random' }
  let(:transformer) { described_class.new(n_components: n_components, init: init, random_seed: 1) }
  let(:z) { transformer.fit_transform(x, y) }

  context 'when n_components is not given' do
    it 'projects data into subspace', :aggregate_failures do
      expect(z).to be_a(Numo::DFloat)
      expect(z).to be_contiguous
      expect(z.ndim).to eq(2)
      expect(z.shape[0]).to eq(n_samples)
      expect(z.shape[1]).to eq(n_features)
      expect(transformer.components).to be_a(Numo::DFloat)
      expect(transformer.components).to be_contiguous
      expect(transformer.components.ndim).to eq(2)
      expect(transformer.components.shape[0]).to eq(n_features)
      expect(transformer.components.shape[1]).to eq(n_features)
      expect(transformer.n_iter).to be_a(Numeric)
    end
  end

  context 'when n_components sets to half of n_features' do
    let(:n_components) { n_features / 2 }
    let(:dataset_ids) { Array(0...n_samples) }
    let(:train_ids) { dataset_ids.sample(n_samples * 0.9, random: Random.new(1984)) }
    let(:test_ids) { dataset_ids - train_ids }
    let(:x_train) { x[train_ids, true].dup }
    let(:x_test) { x[test_ids, true].dup }
    let(:y_train) { y[train_ids].dup }
    let(:y_test) { y[test_ids].dup }
    let(:z_train) { transformer.fit_transform(x_train, y_train) }
    let(:z_test) { transformer.transform(x_test) }
    let(:regressor) { Rumale::NearestNeighbors::KNeighborsRegressor.new(n_neighbors: 5) }

    before { regressor.fit(z_train, y_train) }

    it 'projects data into subspace', :aggregate_failures do
      expect(transformer.components).to be_a(Numo::DFloat)
      expect(transformer.components).to be_contiguous
      expect(transformer.components.ndim).to eq(2)
      expect(transformer.components.shape[0]).to eq(n_components)
      expect(transformer.components.shape[1]).to eq(n_features)
      expect(regressor.score(z_test, y_test)).to be_within(0.05).of(0.7)
    end
  end

  context 'when subspace dimensionality is one' do
    let(:n_components) { 1 }

    it 'projects data into one-dimensional subspace.', :aggregate_failures do
      expect(z).to be_a(Numo::DFloat)
      expect(z).to be_contiguous
      expect(z.ndim).to eq(1)
      expect(z.shape[0]).to eq(n_samples)
      expect(transformer.components).to be_a(Numo::DFloat)
      expect(transformer.components).to be_contiguous
      expect(transformer.components.ndim).to eq(1)
      expect(transformer.components.shape[0]).to eq(n_features)
    end
  end

  context 'when initializing components with PCA' do
    let(:init) { 'pca' }

    before { transformer.fit_transform(x, y) }

    it 'converges more quickly with simple dataset' do
      expect(transformer.n_iter).to be < 5
    end
  end
end
