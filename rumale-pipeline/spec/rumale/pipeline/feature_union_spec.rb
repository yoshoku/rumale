# frozen_string_literal: true

require 'spec_helper'

require 'rumale/kernel_approximation/rbf'
require 'rumale/decomposition/pca'
require 'rumale/decomposition/nmf'

RSpec.describe Rumale::Pipeline::FeatureUnion do
  let(:n_samples) { 32 }
  let(:n_features) { 16 }
  let(:x) { Numo::DFloat.new(n_samples, n_features).rand }
  let(:n_rbf_comps) { 64 }
  let(:n_nmf_comps) { 8 }
  let(:n_pca_comps) { 4 }
  let(:sum_n_comps) { n_rbf_comps + n_pca_comps + n_nmf_comps }
  let(:rbf) { Rumale::KernelApproximation::RBF.new(gamma: 0.1, n_components: n_rbf_comps, random_seed: 1) }
  let(:pca) { Rumale::Decomposition::PCA.new(n_components: n_pca_comps, tol: 1.0e-8, random_seed: 1) }
  let(:nmf) { Rumale::Decomposition::NMF.new(n_components: n_nmf_comps, random_seed: 1) }
  let(:fu) { described_class.new(transformers: { rbf: rbf, pca: pca, nmf: nmf }) }
  let(:z) { fu.fit_transform(x) }

  it 'concatenates each transformed data', :aggregate_failures do
    expect(z.class).to eq(Numo::DFloat)
    expect(z.ndim).to eq(2)
    expect(z.shape[0]).to eq(n_samples)
    expect(z.shape[1]).to eq(sum_n_comps)
  end
end
