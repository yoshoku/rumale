# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Manifold::MDS do
  let(:base_samples) { two_clusters_dataset[0] }
  let(:samples) { Rumale::KernelApproximation::RBF.new(n_components: 32, random_seed: 1).fit_transform(base_samples) }
  let(:n_samples) { samples.shape[0] }
  let(:n_features) { samples.shape[1] }
  let(:n_components) { 2 }
  let(:metric) { 'euclidean' }
  let(:tol) { nil }
  let(:verbose) { false }
  let(:max_iter) { 100 }
  let(:mds) do
    described_class.new(n_components: n_components, metric: metric, max_iter: max_iter, tol: tol, init: 'pca',
                        verbose: verbose, random_seed: 1)
  end
  let(:init_stress) do
    described_class.new(n_components: n_components, metric: metric, max_iter: 0, tol: tol, init: 'pca',
                        verbose: verbose, random_seed: 1).fit(x).stress
  end
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
      expect(mds.n_iter).to eq(max_iter)
      expect(mds.stress.class).to eq(Float)
      expect(mds.stress).not_to be_nil
      expect(mds.stress).to be < init_stress
    end

    context 'when tol parameter is given' do
      let(:tol) { 1000 }

      it 'terminates optimization based on the tol parameter', :aggregate_failures do
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
        expect(mds.n_iter).to be < max_iter
        expect(mds.stress).to be_a(Float)
        expect(mds.stress).not_to be_nil
        expect(mds.stress).to be < init_stress
      end
    end

    context 'when verbose is "true"' do
      let(:verbose) { true }
      let(:max_iter) { 100 }

      it 'outputs debug messages', :aggregate_failures do
        expect { mds.fit(x) }.to output(/MDS/).to_stdout
      end
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
      expect(mds.n_iter).to eq(max_iter)
      expect(mds.stress).to be_a(Float)
      expect(mds.stress).not_to be_nil
      expect(mds.stress).to be < init_stress
    end

    it 'raises ArgumentError when given a non-square matrix', :aggregate_failures do
      expect { mds.fit(Numo::DFloat.new(5, 3).rand) }.to raise_error(ArgumentError)
    end
  end
end
