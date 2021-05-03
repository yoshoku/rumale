# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Decomposition::NMF do
  let(:n_samples) { 200 }
  let(:n_features) { 6 }
  let(:n_components) { 3 }
  let(:decomposer) { described_class.new(n_components: n_components, max_iter: 10000, tol: 1.0e-4, random_seed: 1) }
  let(:x) do
    rng = Random.new(1)
    a = Rumale::Utils.rand_uniform([n_samples, n_components], rng)
    b = Rumale::Utils.rand_uniform([n_components, n_features], rng)
    a.dot(b)
  end

  it 'decomposes non-negative matrix.', :aggregate_failures do
    x_coef = decomposer.fit_transform(x)
    expect(x_coef).to be_a(Numo::DFloat)
    expect(x_coef).to be_contiguous
    expect(x_coef.ndim).to eq(2)
    expect(x_coef.shape[0]).to eq(n_samples)
    expect(x_coef.shape[1]).to eq(n_components)
    expect(decomposer.components).to be_a(Numo::DFloat)
    expect(decomposer.components).to be_contiguous
    expect(decomposer.components.ndim).to eq(2)
    expect(decomposer.components.shape[0]).to eq(n_components)
    expect(decomposer.components.shape[1]).to eq(n_features)
    x_rec = decomposer.inverse_transform(x_coef)
    mse = ((x - x_rec)**2).sum(1).mean
    expect(mse).to be <= 1.0e-4
  end

  it 'transform non-negative coefficients.', :aggregate_failures do
    decomposer.fit(x)
    comp = decomposer.components.dup
    y = x[0...10, true] + Numo::DFloat.new(10, n_features).rand * 0.01
    y_coef = decomposer.transform(y)
    expect(y_coef).to be_a(Numo::DFloat)
    expect(y_coef).to be_contiguous
    expect(y_coef.ndim).to eq(2)
    expect(y_coef.shape[0]).to eq(10)
    expect(y_coef.shape[1]).to eq(n_components)
    expect(decomposer.components).to eq(comp)
    expect(decomposer.components).to be_a(Numo::DFloat)
    expect(decomposer.components).to be_contiguous
    expect(decomposer.components.ndim).to eq(2)
    expect(decomposer.components.shape[0]).to eq(n_components)
    expect(decomposer.components.shape[1]).to eq(n_features)
    y_rec = decomposer.inverse_transform(y_coef)
    mse = ((y - y_rec)**2).sum(1).mean
    expect(mse).to be <= 1.0e-4
  end

  it 'decomposes with one-component factor.' do
    liner = described_class.new(n_components: 1, random_seed: 1)
    sub_x = liner.fit_transform(x)
    rec_x = liner.inverse_transform(sub_x)
    expect(rec_x.shape[0]).to eq(x.shape[0])
    expect(rec_x.shape[1]).to eq(x.shape[1])
    mse = Numo::NMath.sqrt(((x - rec_x)**2).sum(1)).mean
    expect(mse).to be <= 0.5
  end

  it 'dumps and restores itself using Marshal module.' do
    decomposer.fit(x)
    copied = Marshal.load(Marshal.dump(decomposer))
    expect(decomposer.class).to eq(copied.class)
    expect(decomposer.params[:n_components]).to eq(copied.params[:n_components])
    expect(decomposer.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(decomposer.params[:tol]).to eq(copied.params[:tol])
    expect(decomposer.params[:eps]).to eq(copied.params[:eps])
    expect(decomposer.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(decomposer.components).to eq(copied.components)
    expect(decomposer.rng).to eq(copied.rng)
  end
end
