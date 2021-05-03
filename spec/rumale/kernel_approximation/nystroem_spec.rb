# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::KernelApproximation::Nystroem do
  let(:x) { three_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_components) { 64 }
  let(:kernel) { 'rbf' }
  let(:gamma) { 1 }
  let(:degree) { 1 }
  let(:coef) { 1 }
  let(:transformer) do
    described_class.new(kernel: kernel, gamma: gamma, degree: degree, coef: coef, n_components: n_components, random_seed: 1)
  end
  let(:copied) { Marshal.load(Marshal.dump(transformer.fit(x))) }

  shared_examples 'calculating kernel approximiation' do
    let(:mapped_x) { transformer.fit_transform(x) }
    let(:inner_prod) { mapped_x.dot(mapped_x.transpose) }
    let(:mse) { ((kernel_mat - inner_prod)**2).sum.fdiv(n_samples**2) }

    it 'has a small approximation error for the kernel function.', :aggregate_failures do
      expect(mse).to be < 0.01
      expect(mapped_x).to be_a(Numo::DFloat)
      expect(mapped_x).to be_contiguous
      expect(mapped_x.ndim).to eq(2)
      expect(mapped_x.shape[0]).to eq(n_samples)
      expect(mapped_x.shape[1]).to eq(n_components)
      expect(transformer.components).to be_a(Numo::DFloat)
      expect(transformer.components).to be_contiguous
      expect(transformer.components.ndim).to eq(2)
      expect(transformer.components.shape[0]).to eq(n_components)
      expect(transformer.components.shape[1]).to eq(n_features)
      expect(transformer.component_indices).to be_a(Numo::Int32)
      expect(transformer.component_indices).to be_contiguous
      expect(transformer.component_indices.ndim).to eq(1)
      expect(transformer.component_indices.shape[0]).to eq(n_components)
      expect(transformer.normalizer).to be_a(Numo::DFloat)
      expect(transformer.normalizer).to be_contiguous
      expect(transformer.normalizer.ndim).to eq(2)
      expect(transformer.normalizer.shape[0]).to eq(n_components)
      expect(transformer.normalizer.shape[1]).to eq(n_components)
    end
  end

  context "when kernel is 'rbf'" do
    let(:kernel) { 'rbf' }
    let(:gamma) { 0.5 }
    let(:kernel_mat) { Rumale::PairwiseMetric.rbf_kernel(x, nil, gamma) }

    it_behaves_like 'calculating kernel approximiation'
  end

  context "when kernel is 'poly'" do
    let(:kernel) { 'poly' }
    let(:degree) { 3 }
    let(:kernel_mat) { Rumale::PairwiseMetric.polynomial_kernel(x, nil, degree, gamma, coef) }

    it_behaves_like 'calculating kernel approximiation'
  end

  context "when kernel is 'sigmoid'" do
    let(:kernel) { 'sigmoid' }
    let(:gamma) { 1e-5 }
    let(:coef) { 1e-1 }
    let(:kernel_mat) { Rumale::PairwiseMetric.sigmoid_kernel(x, nil, gamma, coef) }

    it_behaves_like 'calculating kernel approximiation'
  end

  context "when kernel is 'linear'" do
    let(:kernel) { 'linear' }
    let(:kernel_mat) { Rumale::PairwiseMetric.linear_kernel(x) }

    it_behaves_like 'calculating kernel approximiation'
  end

  context "when kernel is 'foo'" do
    subject(:transform) { transformer.fit_transform(x) }

    let(:kernel) { 'foo' }

    it 'raises ArgumentError' do
      expect { transform }.to raise_error(ArgumentError)
    end
  end

  it 'dumps and restores itself using Marshal module.', :aggregate_failures do
    expect(transformer.class).to eq(copied.class)
    expect(transformer.params).to eq(copied.params)
    expect(transformer.components).to eq(copied.components)
    expect(transformer.component_indices).to eq(copied.component_indices)
    expect(transformer.normalizer).to eq(copied.normalizer)
    expect(transformer.rng).to eq(copied.rng)
  end
end
