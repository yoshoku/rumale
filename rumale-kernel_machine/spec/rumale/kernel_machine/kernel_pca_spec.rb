# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::KernelMachine::KernelPCA do
  let(:n_samples) { x.shape[0] }
  let(:n_components) { 2 }
  let(:transformer) { described_class.new(n_components: n_components) }
  let(:dataset_ids) { Array(0...n_samples) }
  let(:train_ids) { dataset_ids.sample(n_samples * 0.9, random: Random.new(1984)) }
  let(:test_ids) { dataset_ids - train_ids }
  let(:x_train) { x[train_ids, true].dup }
  let(:x_test) { x[test_ids, true].dup }
  let(:y_train) { y[train_ids].dup }
  let(:y_test) { y[test_ids].dup }
  let(:n_train_samples) { x_train.shape[0] }
  let(:n_test_samples) { x_test.shape[0] }

  describe 'basic examples' do
    let(:two_clusters) { two_clusters_dataset }
    let(:x) { two_clusters[0] }
    let(:y) { two_clusters[1] }
    let(:kernel_mat_train) { Rumale::PairwiseMetric.linear_kernel(x_train, nil) }
    let(:kernel_mat_test) { Rumale::PairwiseMetric.linear_kernel(x_test, x_train) }
    let(:z_train) { transformer.fit_transform(kernel_mat_train) }
    let(:z_test) { transformer.transform(kernel_mat_test) }

    it 'maps into subspace', :aggregate_failures do
      expect(z_train).to be_a(Numo::DFloat)
      expect(z_train).to be_contiguous
      expect(z_train.ndim).to eq(2)
      expect(z_train.shape[0]).to eq(n_train_samples)
      expect(z_train.shape[1]).to eq(n_components)
      expect(z_test).to be_a(Numo::DFloat)
      expect(z_test).to be_contiguous
      expect(z_test.ndim).to eq(2)
      expect(z_test.shape[0]).to eq(n_test_samples)
      expect(z_test.shape[1]).to eq(n_components)
      expect(transformer.alphas).to be_a(Numo::DFloat)
      expect(transformer.alphas).to be_contiguous
      expect(transformer.alphas.ndim).to eq(2)
      expect(transformer.alphas.shape[0]).to eq(n_train_samples)
      expect(transformer.alphas.shape[1]).to eq(n_components)
      expect(transformer.lambdas).to be_a(Numo::DFloat)
      expect(transformer.lambdas).to be_contiguous
      expect(transformer.lambdas.ndim).to eq(1)
      expect(transformer.lambdas.shape[0]).to eq(n_components)
    end

    context 'when one-dimensional subspace' do
      let(:n_components) { 1 }
      let(:pca) { Rumale::Decomposition::PCA.new(n_components: n_components, solver: 'evd').fit(x_train) }
      let(:z_train_pca) { pca.transform(x_train) }
      let(:z_test_pca) { pca.transform(x_test) }

      it 'maps data into one-dimensional subspace', :aggregate_failures do
        expect(z_train).to be_a(Numo::DFloat)
        expect(z_train).to be_contiguous
        expect(z_train.ndim).to eq(1)
        expect(z_train.shape[0]).to eq(n_train_samples)
        expect(z_test).to be_a(Numo::DFloat)
        expect(z_test).to be_contiguous
        expect(z_test.ndim).to eq(1)
        expect(z_test.shape[0]).to eq(n_test_samples)
      end

      it 'has small error from PCA in the case of a linear kernel', :aggregate_failures do
        expect(Math.sqrt(((z_train - z_train_pca)**2).sum)).to be < 1e-8
        expect(Math.sqrt(((z_test - z_test_pca)**2).sum)).to be < 1e-8
      end
    end
  end

  describe 'using with linear classifier' do
    let(:xor) { xor_dataset }
    let(:x) { xor[0] }
    let(:y) { xor[1] }
    let(:kernel_mat_train) { Rumale::PairwiseMetric.rbf_kernel(x_train, nil, 1.0) }
    let(:kernel_mat_test) { Rumale::PairwiseMetric.rbf_kernel(x_test, x_train, 1.0) }
    let(:z_train) { transformer.fit_transform(kernel_mat_train) }
    let(:z_test) { transformer.transform(kernel_mat_test) }
    let(:classifier) { Rumale::LinearModel::SVC.new(reg_param: 0.01, fit_bias: false) }

    before { classifier.fit(z_train, y_train) }

    it 'maps to a linearly separable space', :aggregate_failures do
      expect(z_train).to be_a(Numo::DFloat)
      expect(z_train).to be_contiguous
      expect(z_train.ndim).to eq(2)
      expect(z_train.shape[0]).to eq(n_train_samples)
      expect(z_train.shape[1]).to eq(n_components)
      expect(z_test).to be_a(Numo::DFloat)
      expect(z_test).to be_contiguous
      expect(z_test.ndim).to eq(2)
      expect(z_test.shape[0]).to eq(n_test_samples)
      expect(z_test.shape[1]).to eq(n_components)
      expect(classifier.score(z_train, y_train)).to be_within(0.01).of(1.0)
      expect(classifier.score(z_test, y_test)).to be_within(0.01).of(1.0)
    end
  end
end
