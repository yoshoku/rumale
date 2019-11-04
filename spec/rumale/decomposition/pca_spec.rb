# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Decomposition::PCA do
  let(:x) { two_clusters_dataset[0] }
  let(:n_components) { 16 }
  let(:solver) { 'fpt' }
  let(:decomposer) { described_class.new(n_components: n_components, solver: solver, tol: 1.0e-8, random_seed: 1) }
  let(:transformer) { Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_components: 32, random_seed: 1) }
  let(:samples) { transformer.fit_transform(x) }
  let(:sub_samples) { decomposer.fit_transform(samples) }
  let(:rec_samples) { decomposer.inverse_transform(sub_samples) }
  let(:mse) { Numo::NMath.sqrt(((samples - rec_samples)**2).sum(1)).mean }
  let(:n_samples) { samples.shape[0] }
  let(:n_features) { samples.shape[1] }

  shared_examples 'projection into subspace' do
    it 'projects high-dimensinal data into subspace.' do
      expect(sub_samples.class).to eq(Numo::DFloat)
      expect(sub_samples.shape[0]).to eq(n_samples)
      expect(sub_samples.shape[1]).to eq(n_components)
      expect(rec_samples.shape[0]).to eq(n_samples)
      expect(rec_samples.shape[1]).to eq(n_features)
      expect(decomposer.components.class).to eq(Numo::DFloat)
      expect(decomposer.components.shape[0]).to eq(n_components)
      expect(decomposer.components.shape[1]).to eq(n_features)
      expect(decomposer.mean.class).to eq(Numo::DFloat)
      expect(decomposer.mean.shape[0]).to eq(n_features)
      expect(decomposer.mean.shape[1]).to be_nil
      expect(mse).to be <= 0.1
    end
  end

  shared_examples 'projection into one-dimensional subspace' do
    let(:n_components) { 1 }
    let(:samples) { x }
    let(:sub_samples) { decomposer.fit_transform(samples).expand_dims(1) }

    it 'projects data into one-dimensional subspace.' do
      expect(sub_samples.shape[0]).to eq(n_samples)
      expect(sub_samples.shape[1]).to eq(n_components)
      expect(rec_samples.shape[0]).to eq(n_samples)
      expect(rec_samples.shape[1]).to eq(n_features)
      expect(decomposer.components.shape[0]).to eq(n_features)
      expect(decomposer.components.shape[1]).to be_nil
      expect(decomposer.mean.shape[0]).to eq(n_features)
      expect(decomposer.mean.shape[1]).to be_nil
      expect(mse).to be <= 0.5
    end
  end

  context 'when solver is fix point algorithm' do
    it_behaves_like 'projection into subspace'
    it_behaves_like 'projection into one-dimensional subspace'
  end

  context 'when solver is eigen value decomposition' do
    let(:solver) { 'evd' }

    it_behaves_like 'projection into subspace'
    it_behaves_like 'projection into one-dimensional subspace'
  end

  it 'dumps and restores itself using Marshal module.' do
    copied = Marshal.load(Marshal.dump(decomposer))
    expect(decomposer.class).to eq(copied.class)
    expect(decomposer.params[:n_components]).to eq(copied.params[:n_components])
    expect(decomposer.params[:solver]).to eq(copied.params[:solver])
    expect(decomposer.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(decomposer.params[:tol]).to eq(copied.params[:tol])
    expect(decomposer.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(decomposer.components).to eq(copied.components)
    expect(decomposer.mean).to eq(copied.mean)
    expect(decomposer.rng).to eq(copied.rng)
  end
end
