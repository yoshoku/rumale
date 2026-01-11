# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Manifold::ClassicalMDS do
  let(:base_samples) { two_clusters_dataset[0] }
  let(:samples) { Rumale::KernelApproximation::RBF.new(n_components: 32, random_seed: 1).fit_transform(base_samples) }
  let(:n_samples) { samples.shape[0] }
  let(:n_features) { samples.shape[1] }
  let(:n_components) { 2 }
  let(:metric) { 'euclidean' }
  let(:mds) { described_class.new(n_components: n_components, metric: metric) }
  let(:low_samples) { mds.fit_transform(x) }

  context 'when metric is "euclidean"' do
    let(:metric) { 'euclidean' }
    let(:x) { samples }

    it 'maps high-dimensional data into low-dimensional data', :aggregate_failures do
      expect(low_samples).to be_a(Numo::DFloat)
      expect(low_samples).to be_contiguous
      expect(low_samples.ndim).to eq(2)
      expect(low_samples.shape[0]).to eq(n_samples)
      expect(low_samples.shape[1]).to eq(n_components)
      expect(mds.embedding).to be_a(Numo::DFloat)
      expect(mds.embedding).to be_contiguous
      expect(mds.embedding.ndim).to eq(2)
      expect(mds.embedding.shape[0]).to eq(n_samples)
      expect(mds.embedding.shape[1]).to eq(n_components)
    end
  end

  context 'when metric is "precomputed"' do
    let(:metric) { 'precomputed' }
    let(:x) { Rumale::PairwiseMetric.euclidean_distance(samples) }

    it 'maps high-dimensional data represented by distance matrix', :aggregate_failures do
      expect(low_samples).to be_a(Numo::DFloat)
      expect(low_samples).to be_contiguous
      expect(low_samples.ndim).to eq(2)
      expect(low_samples.shape[0]).to eq(n_samples)
      expect(low_samples.shape[1]).to eq(n_components)
      expect(mds.embedding).to be_a(Numo::DFloat)
      expect(mds.embedding).to be_contiguous
      expect(mds.embedding.ndim).to eq(2)
      expect(mds.embedding.shape[0]).to eq(n_samples)
      expect(mds.embedding.shape[1]).to eq(n_components)
    end

    it 'raises ArgumentError when given a non-square matrix', :aggregate_failures do
      expect { mds.fit(Numo::DFloat.new(5, 3).rand) }.to raise_error(ArgumentError)
    end
  end
end
