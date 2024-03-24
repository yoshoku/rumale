# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Decomposition::SparsePCA do
  let(:x) { two_clusters_dataset[0] }
  let(:n_components) { 4 }
  let(:decomposer) { described_class.new(n_components: n_components, reg_param: 0.001, random_seed: 1984) }
  let(:samples) { Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_components: 16, random_seed: 1984).fit_transform(x) }
  let(:sub_samples) { decomposer.fit_transform(samples) }
  let(:n_samples) { samples.shape[0] }
  let(:n_features) { samples.shape[1] }

  shared_examples 'projection into subspace' do
    it 'projects high-dimensinal data into subspace', :aggregate_failures do
      expect(sub_samples).to be_a(Numo::DFloat)
      expect(sub_samples).to be_contiguous
      expect(sub_samples.ndim).to eq(2)
      expect(sub_samples.shape[0]).to eq(n_samples)
      expect(sub_samples.shape[1]).to eq(n_components)
      expect(decomposer.components).to be_a(Numo::DFloat)
      expect(decomposer.components).to be_contiguous
      expect(decomposer.components.ndim).to eq(2)
      expect(decomposer.components.shape[0]).to eq(n_components)
      expect(decomposer.components.shape[1]).to eq(n_features)
      expect(decomposer.mean).to be_a(Numo::DFloat)
      expect(decomposer.mean).to be_contiguous
      expect(decomposer.mean.ndim).to eq(1)
      expect(decomposer.mean.shape[0]).to eq(n_features)
    end
  end

  shared_examples 'projection into one-dimensional subspace' do
    let(:n_components) { 1 }
    let(:samples) { x }
    let(:sub_samples) { decomposer.fit_transform(samples).expand_dims(1).dup }

    it 'projects data into one-dimensional subspace', :aggregate_failures do
      expect(sub_samples).to be_a(Numo::DFloat)
      expect(sub_samples).to be_contiguous
      expect(sub_samples.ndim).to eq(2)
      expect(sub_samples.shape[0]).to eq(n_samples)
      expect(sub_samples.shape[1]).to eq(n_components)
      expect(decomposer.components).to be_a(Numo::DFloat)
      expect(decomposer.components).to be_contiguous
      expect(decomposer.components.ndim).to eq(1)
      expect(decomposer.components.shape[0]).to eq(n_features)
      expect(decomposer.mean).to be_a(Numo::DFloat)
      expect(decomposer.mean).to be_contiguous
      expect(decomposer.mean.ndim).to eq(1)
      expect(decomposer.mean.shape[0]).to eq(n_features)
    end
  end

  it_behaves_like 'projection into subspace'
  it_behaves_like 'projection into one-dimensional subspace'
end
