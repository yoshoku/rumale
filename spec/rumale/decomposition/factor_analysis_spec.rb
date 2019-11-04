# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Decomposition::FactorAnalysis do
  let(:x) { two_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:n_components) { x.shape[1] }
  let(:n_features) { 8 }
  let(:samples) do
    w = Rumale::Utils.rand_normal([n_components, n_features], Random.new(1))
    x.dot(w) + 0.001 * Rumale::Utils.rand_normal([n_samples, n_features], Random.new(1))
  end
  let(:max_iter) { 500 }
  let(:tol) { 1e-8 }
  let(:decomposer) { described_class.new(n_components: n_components, max_iter: max_iter, tol: tol) }
  let(:sub_samples) { decomposer.fit_transform(samples) }
  let(:error) do
    centered = samples - samples.mean(0)
    cov_mat = centered.transpose.dot(centered) / n_samples
    fact_cov_mat = decomposer.components.transpose.dot(decomposer.components) + decomposer.noise_variance.diag
    (cov_mat - fact_cov_mat).abs.mean
  end

  shared_examples 'projection into subspace' do
    it 'projects high-dimensinal data into subspace.', aggregate_failures: true do
      expect(sub_samples.class).to eq(Numo::DFloat)
      expect(sub_samples.shape[0]).to eq(n_samples)
      expect(sub_samples.shape[1]).to eq(n_components)
      expect(decomposer.mean.class).to eq(Numo::DFloat)
      expect(decomposer.mean.shape[0]).to eq(n_features)
      expect(decomposer.mean.shape[1]).to be_nil
      expect(decomposer.noise_variance.class).to eq(Numo::DFloat)
      expect(decomposer.noise_variance.shape[0]).to eq(n_features)
      expect(decomposer.noise_variance.shape[1]).to be_nil
      expect(decomposer.components.class).to eq(Numo::DFloat)
      expect(decomposer.components.shape[0]).to eq(n_components)
      expect(decomposer.components.shape[1]).to eq(n_features)
      expect(decomposer.n_iter).to be < max_iter
      expect(decomposer.loglike.class).to eq(Numo::DFloat)
      expect(decomposer.loglike.shape[0]).to eq(decomposer.n_iter)
      expect(decomposer.loglike.shape[1]).to be_nil
      expect(error).to be <= 0.1
    end
  end

  shared_examples 'projection into one-dimensional subspace' do
    let(:n_features) { x.shape[1] }
    let(:n_components) { 1 }
    let(:samples) { x }
    let(:tol) { nil }
    let(:sub_samples) { decomposer.fit_transform(samples).expand_dims(1) }

    it 'projects data into one-dimensional subspace.', aggregate_failures: true do
      expect(sub_samples.shape[0]).to eq(n_samples)
      expect(sub_samples.shape[1]).to eq(n_components)
      expect(decomposer.mean.class).to eq(Numo::DFloat)
      expect(decomposer.mean.shape[0]).to eq(n_features)
      expect(decomposer.mean.shape[1]).to be_nil
      expect(decomposer.noise_variance.class).to eq(Numo::DFloat)
      expect(decomposer.noise_variance.shape[0]).to eq(n_features)
      expect(decomposer.noise_variance.shape[1]).to be_nil
      expect(decomposer.components.class).to eq(Numo::DFloat)
      expect(decomposer.components.shape[0]).to eq(n_features)
      expect(decomposer.components.shape[1]).to be_nil
      expect(decomposer.n_iter).to eq(max_iter)
      expect(decomposer.loglike).to be_nil
    end
  end

  context 'when solver is fix point algorithm' do
    it_behaves_like 'projection into subspace'
    it_behaves_like 'projection into one-dimensional subspace'
  end

  it 'dumps and restores itself using Marshal module.', aggregate_failures: true do
    copied = Marshal.load(Marshal.dump(decomposer))
    expect(decomposer.class).to eq(copied.class)
    expect(decomposer.params[:n_components]).to eq(copied.params[:n_components])
    expect(decomposer.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(decomposer.params[:tol]).to eq(copied.params[:tol])
    expect(decomposer.mean).to eq(copied.mean)
    expect(decomposer.noise_variance).to eq(copied.noise_variance)
    expect(decomposer.components).to eq(copied.components)
    expect(decomposer.n_iter).to eq(copied.n_iter)
    expect(decomposer.loglike).to eq(copied.loglike)
  end
end
