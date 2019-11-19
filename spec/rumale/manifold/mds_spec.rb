# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Manifold::MDS do
  let(:samples) do
    Rumale::KernelApproximation::RBF.new(n_components: 32, random_seed: 1).fit_transform(two_clusters_dataset[0])
  end
  let(:n_samples) { samples.shape[0] }
  let(:n_features) { samples.shape[1] }
  let(:n_components) { 2 }
  let(:metric) { 'euclidean' }
  let(:tol) { nil }
  let(:verbose) { false }
  let(:max_iter) { 50 }
  let(:mds) do
    described_class.new(n_components: n_components, metric: metric, max_iter: max_iter, tol: tol, init: 'pca',
                        verbose: verbose, random_seed: 1)
  end
  let(:init_stress) do
    described_class.new(n_components: n_components, metric: metric, max_iter: 0, tol: tol, init: 'pca',
                        verbose: verbose, random_seed: 1).fit(x).stress
  end
  let(:low_samples) { mds.fit_transform(x) }
  let(:copied) { Marshal.load(Marshal.dump(mds)) }

  context 'when metric is "euclidean"' do
    let(:metric) { 'euclidean' }
    let(:x) { samples }

    it 'maps high-dimensional data into low-dimensional data.', :aggregate_failures do
      expect(low_samples.class).to eq(Numo::DFloat)
      expect(low_samples.shape[0]).to eq(n_samples)
      expect(low_samples.shape[1]).to eq(n_components)
      expect(mds.embedding.class).to eq(Numo::DFloat)
      expect(mds.embedding.shape[0]).to eq(n_samples)
      expect(mds.embedding.shape[1]).to eq(n_components)
      expect(mds.n_iter).to eq(max_iter)
      expect(mds.stress.class).to eq(Float)
      expect(mds.stress).not_to be_nil
      expect(mds.stress).to be < init_stress
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(mds.class).to eq(copied.class)
      expect(mds.params[:n_components]).to match(copied.params[:n_components])
      expect(mds.params[:max_iter]).to match(copied.params[:max_iter])
      expect(mds.params[:metric]).to match(copied.params[:metric])
      expect(mds.params[:init]).to match(copied.params[:init])
      expect(mds.params[:verbose]).to match(copied.params[:verbose])
      expect(mds.params[:random_seed]).to match(copied.params[:random_seed])
      expect(mds.embedding).to eq(copied.embedding)
      expect(mds.stress).to eq(copied.stress)
      expect(mds.n_iter).to eq(copied.n_iter)
      expect(mds.rng).to eq(copied.rng)
    end

    context 'when tol parameter is given' do
      let(:tol) { 1000 }

      it 'terminates optimization based on the tol parameter.', :aggregate_failures do
        expect(low_samples.class).to eq(Numo::DFloat)
        expect(low_samples.shape[0]).to eq(n_samples)
        expect(low_samples.shape[1]).to eq(n_components)
        expect(mds.embedding.class).to eq(Numo::DFloat)
        expect(mds.embedding.shape[0]).to eq(n_samples)
        expect(mds.embedding.shape[1]).to eq(n_components)
        expect(mds.n_iter).to be < max_iter
        expect(mds.stress.class).to eq(Float)
        expect(mds.stress).not_to be_nil
        expect(mds.stress).to be < init_stress
      end
    end

    context 'when verbose is "true"' do
      let(:verbose) { true }
      let(:max_iter) { 100 }

      it 'outputs debug messages.', :aggregate_failures do
        expect { mds.fit(x) }.to output(/MDS/).to_stdout
      end
    end
  end

  context 'when metric is "precomputed"' do
    let(:metric) { 'precomputed' }
    let(:x) { Rumale::PairwiseMetric.euclidean_distance(samples) }

    it 'maps high-dimensional data represented by distance matrix.', :aggregate_failures do
      expect(low_samples.class).to eq(Numo::DFloat)
      expect(low_samples.shape[0]).to eq(n_samples)
      expect(low_samples.shape[1]).to eq(n_components)
      expect(mds.embedding.class).to eq(Numo::DFloat)
      expect(mds.embedding.shape[0]).to eq(n_samples)
      expect(mds.embedding.shape[1]).to eq(n_components)
      expect(mds.n_iter).to eq(50)
      expect(mds.stress.class).to eq(Float)
      expect(mds.stress).not_to be_nil
      expect(mds.stress).to be < init_stress
    end

    it 'raises ArgumentError when given a non-square matrix.', :aggregate_failures do
      expect { mds.fit(Numo::DFloat.new(5, 3).rand) }.to raise_error(ArgumentError)
    end
  end
end
