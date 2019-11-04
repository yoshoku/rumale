# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Manifold::TSNE do
  let(:x) { two_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { 32 }
  let(:n_components) { 2 }
  let(:samples) { Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_components: n_features, random_seed: 1).fit_transform(x) }
  let(:distance_matrix) { Rumale::PairwiseMetric.euclidean_distance(samples) }
  let(:init_kl) { described_class.new(n_components: n_components, max_iter: 0, random_seed: 1).fit(samples).kl_divergence }
  let(:tsne) { described_class.new(n_components: n_components, max_iter: 50, random_seed: 1) }
  let(:tsne_metric) { described_class.new(n_components: n_components, metric: 'precomputed', max_iter: 50, random_seed: 1) }
  let(:tsne_tol) { described_class.new(n_components: n_components, max_iter: 50, tol: 1.0, random_seed: 1) }
  let(:tsne_params) do
    described_class.new(n_components: n_components, perplexity: 200.0, max_iter: 200,
                        init: 'pca', verbose: true, random_seed: 1)
  end

  it 'maps high-dimensional data into low-dimensional data.' do
    low_samples = tsne.fit_transform(samples)
    expect(low_samples.class).to eq(Numo::DFloat)
    expect(low_samples.shape[0]).to eq(n_samples)
    expect(low_samples.shape[1]).to eq(n_components)
    expect(tsne.embedding.class).to eq(Numo::DFloat)
    expect(tsne.embedding.shape[0]).to eq(n_samples)
    expect(tsne.embedding.shape[1]).to eq(n_components)
    expect(tsne.n_iter).to eq(50)
    expect(tsne.kl_divergence.class).to eq(Float)
    expect(tsne.kl_divergence).not_to be_nil
    expect(tsne.kl_divergence).to be < init_kl
  end

  it 'maps high-dimensional data represented by distance matrix.' do
    low_samples = tsne_metric.fit_transform(distance_matrix)
    expect(low_samples.class).to eq(Numo::DFloat)
    expect(low_samples.shape[0]).to eq(n_samples)
    expect(low_samples.shape[1]).to eq(n_components)
    expect(tsne_metric.embedding.class).to eq(Numo::DFloat)
    expect(tsne_metric.embedding.shape[0]).to eq(n_samples)
    expect(tsne_metric.embedding.shape[1]).to eq(n_components)
    expect(tsne_metric.n_iter).to eq(50)
    expect(tsne_metric.kl_divergence.class).to eq(Float)
    expect(tsne_metric.kl_divergence).not_to be_nil
    expect(tsne_metric.kl_divergence).to be < init_kl
  end

  it 'terminates optimization based on the tol parameter.' do
    low_samples = tsne_tol.fit_transform(samples)
    expect(low_samples.class).to eq(Numo::DFloat)
    expect(low_samples.shape[0]).to eq(n_samples)
    expect(low_samples.shape[1]).to eq(n_components)
    expect(tsne_tol.embedding.class).to eq(Numo::DFloat)
    expect(tsne_tol.embedding.shape[0]).to eq(n_samples)
    expect(tsne_tol.embedding.shape[1]).to eq(n_components)
    expect(tsne_tol.n_iter).to be < 50
    expect(tsne_tol.kl_divergence.class).to eq(Float)
    expect(tsne_tol.kl_divergence).not_to be_nil
    expect(tsne_tol.kl_divergence).to be < init_kl
  end

  it 'raises ArgumentError when given a non-square matrix.' do
    expect { tsne_metric.fit(Numo::DFloat.new(5, 3).rand) }.to raise_error(ArgumentError)
  end

  it 'outputs debug messages.' do
    expect { tsne_params.fit(samples) }.to output(/t-SNE/).to_stdout
  end

  it 'dumps and restores itself using Marshal module.' do
    copied = Marshal.load(Marshal.dump(tsne_params))
    expect(tsne_params.class).to eq(copied.class)
    expect(tsne_params.params[:n_components]).to match(copied.params[:n_components])
    expect(tsne_params.params[:perplexity]).to match(copied.params[:perplexity])
    expect(tsne_params.params[:max_iter]).to match(copied.params[:max_iter])
    expect(tsne_params.params[:metric]).to match(copied.params[:metric])
    expect(tsne_params.params[:init]).to match(copied.params[:init])
    expect(tsne_params.params[:verbose]).to match(copied.params[:verbose])
    expect(tsne_params.params[:random_seed]).to match(copied.params[:random_seed])
    expect(tsne_params.embedding).to eq(copied.embedding)
    expect(tsne_params.kl_divergence).to eq(copied.kl_divergence)
    expect(tsne_params.n_iter).to eq(copied.n_iter)
    expect(tsne_params.rng).to eq(copied.rng)
  end
end
