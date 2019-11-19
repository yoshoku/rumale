# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Manifold::TSNE do
  let(:samples) do
    Rumale::KernelApproximation::RBF.new(n_components: 32, random_seed: 1).fit_transform(two_clusters_dataset[0])
  end
  let(:n_samples) { samples.shape[0] }
  let(:n_features) { samples.shape[1] }
  let(:n_components) { 2 }
  let(:metric) { 'euclidean' }
  let(:perplexity) { 30 }
  let(:tol) { nil }
  let(:verbose) { false }
  let(:max_iter) { 50 }
  let(:tsne) do
    described_class.new(n_components: n_components, metric: metric, max_iter: max_iter, tol: tol, init: 'pca',
                        perplexity: perplexity, verbose: verbose, random_seed: 1)
  end
  let(:init_kl) do
    described_class.new(n_components: n_components, metric: metric, max_iter: 0, tol: tol, init: 'pca',
                        perplexity: perplexity, verbose: verbose, random_seed: 1).fit(x).kl_divergence
  end
  let(:low_samples) { tsne.fit_transform(x) }
  let(:copied) { Marshal.load(Marshal.dump(tsne)) }

  context 'when metric is "euclidean"' do
    let(:metric) { 'euclidean' }
    let(:x) { samples }

    it 'maps high-dimensional data into low-dimensional data.', :aggregate_failures do
      expect(low_samples.class).to eq(Numo::DFloat)
      expect(low_samples.shape[0]).to eq(n_samples)
      expect(low_samples.shape[1]).to eq(n_components)
      expect(tsne.embedding.class).to eq(Numo::DFloat)
      expect(tsne.embedding.shape[0]).to eq(n_samples)
      expect(tsne.embedding.shape[1]).to eq(n_components)
      expect(tsne.n_iter).to eq(max_iter)
      expect(tsne.kl_divergence.class).to eq(Float)
      expect(tsne.kl_divergence).not_to be_nil
      expect(tsne.kl_divergence).to be < init_kl
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(tsne.params[:n_components]).to match(copied.params[:n_components])
      expect(tsne.params[:perplexity]).to match(copied.params[:perplexity])
      expect(tsne.params[:max_iter]).to match(copied.params[:max_iter])
      expect(tsne.params[:metric]).to match(copied.params[:metric])
      expect(tsne.params[:init]).to match(copied.params[:init])
      expect(tsne.params[:verbose]).to match(copied.params[:verbose])
      expect(tsne.params[:random_seed]).to match(copied.params[:random_seed])
      expect(tsne.embedding).to eq(copied.embedding)
      expect(tsne.kl_divergence).to eq(copied.kl_divergence)
      expect(tsne.n_iter).to eq(copied.n_iter)
      expect(tsne.rng).to eq(copied.rng)
    end

    context 'when tol parameter is given' do
      let(:tol) { 1 }

      it 'terminates optimization based on the tol parameter.', :aggregate_failures do
        expect(low_samples.class).to eq(Numo::DFloat)
        expect(low_samples.shape[0]).to eq(n_samples)
        expect(low_samples.shape[1]).to eq(n_components)
        expect(tsne.embedding.class).to eq(Numo::DFloat)
        expect(tsne.embedding.shape[0]).to eq(n_samples)
        expect(tsne.embedding.shape[1]).to eq(n_components)
        expect(tsne.n_iter).to be < max_iter
        expect(tsne.kl_divergence.class).to eq(Float)
        expect(tsne.kl_divergence).not_to be_nil
        expect(tsne.kl_divergence).to be < init_kl
      end
    end

    context 'when verbose is "true"' do
      let(:verbose) { true }
      let(:perplexity) { 200 }
      let(:max_iter) { 100 }

      it 'outputs debug messages.', :aggregate_failures do
        expect { tsne.fit(x) }.to output(/t-SNE/).to_stdout
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
      expect(tsne.embedding.class).to eq(Numo::DFloat)
      expect(tsne.embedding.shape[0]).to eq(n_samples)
      expect(tsne.embedding.shape[1]).to eq(n_components)
      expect(tsne.n_iter).to eq(max_iter)
      expect(tsne.kl_divergence.class).to eq(Float)
      expect(tsne.kl_divergence).not_to be_nil
      expect(tsne.kl_divergence).to be < init_kl
    end

    it 'raises ArgumentError when given a non-square matrix.', :aggregate_failures do
      expect { tsne.fit(Numo::DFloat.new(5, 3).rand) }.to raise_error(ArgumentError)
    end
  end
end
