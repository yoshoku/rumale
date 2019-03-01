# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Decomposition::PCA do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:n_components) { 16 }
  let(:decomposer) { described_class.new(n_components: n_components, tol: 1.0e-8, random_seed: 1) }
  let(:transformer) { Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_components: 32, random_seed: 1) }

  it 'projects high-dimensinal data into subspace.' do
    samples = transformer.fit_transform(x)
    sub_samples = decomposer.fit_transform(samples)

    n_samples, n_features = samples.shape
    expect(sub_samples.class).to eq(Numo::DFloat)
    expect(sub_samples.shape[0]).to eq(n_samples)
    expect(sub_samples.shape[1]).to eq(n_components)

    expect(decomposer.components.class).to eq(Numo::DFloat)
    expect(decomposer.components.shape[0]).to eq(n_components)
    expect(decomposer.components.shape[1]).to eq(n_features)
    expect(decomposer.mean.class).to eq(Numo::DFloat)
    expect(decomposer.mean.shape[0]).to eq(n_features)
    expect(decomposer.mean.shape[1]).to be_nil

    rec_samples = decomposer.inverse_transform(sub_samples)
    mse = Numo::NMath.sqrt(((samples - rec_samples)**2).sum(1)).mean
    expect(mse).to be <= 0.1
  end

  it 'projects data into one-dimensional subspace.' do
    liner = described_class.new(n_components: 1, random_seed: 1)
    sub_x = liner.fit_transform(x)
    rec_x = liner.inverse_transform(sub_x.expand_dims(1))
    expect(rec_x.shape[0]).to eq(x.shape[0])
    expect(rec_x.shape[1]).to eq(x.shape[1])
    mse = Numo::NMath.sqrt(((x - rec_x)**2).sum(1)).mean
    expect(mse).to be <= 0.5
  end

  it 'dumps and restores itself using Marshal module.' do
    samples = transformer.fit_transform(x)
    decomposer.fit(samples)
    copied = Marshal.load(Marshal.dump(decomposer))
    expect(decomposer.class).to eq(copied.class)
    expect(decomposer.params[:n_components]).to eq(copied.params[:n_components])
    expect(decomposer.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(decomposer.params[:tol]).to eq(copied.params[:tol])
    expect(decomposer.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(decomposer.components).to eq(copied.components)
    expect(decomposer.mean).to eq(copied.mean)
    expect(decomposer.rng).to eq(copied.rng)
  end
end
