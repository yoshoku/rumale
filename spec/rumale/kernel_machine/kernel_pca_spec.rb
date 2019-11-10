# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::KernelMachine::KernelPCA do
  let(:n_components) { 2 }
  let(:transformer) { described_class.new(n_components: n_components) }
  let(:splitter) { Rumale::ModelSelection::ShuffleSplit.new(n_splits: 1, test_size: 0.1, train_size: 0.9, random_seed: 1) }
  let(:validation_ids) { splitter.split(x, y).first }
  let(:train_ids) { validation_ids[0] }
  let(:test_ids) { validation_ids[1] }
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
    let(:copied) { Marshal.load(Marshal.dump(transformer.fit(kernel_mat_train))) }

    it 'maps into subspace.', :aggregate_failures do
      expect(z_train.class).to eq(Numo::DFloat)
      expect(z_train.shape[0]).to eq(n_train_samples)
      expect(z_train.shape[1]).to eq(n_components)
      expect(z_test.class).to eq(Numo::DFloat)
      expect(z_test.shape[0]).to eq(n_test_samples)
      expect(z_test.shape[1]).to eq(n_components)
      expect(transformer.alphas.class).to eq(Numo::DFloat)
      expect(transformer.alphas.shape[0]).to eq(n_train_samples)
      expect(transformer.alphas.shape[1]).to eq(n_components)
      expect(transformer.lambdas.class).to eq(Numo::DFloat)
      expect(transformer.lambdas.shape[0]).to eq(n_components)
      expect(transformer.lambdas.shape[1]).to eq(nil)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(transformer.class).to eq(copied.class)
      expect(transformer.params[:n_components]).to eq(copied.params[:n_components])
      expect(transformer.alphas).to eq(copied.alphas)
      expect(transformer.lambdas).to eq(copied.lambdas)
      expect(transformer.instance_variable_get(:@row_mean)).to eq(copied.instance_variable_get(:@row_mean))
      expect(transformer.instance_variable_get(:@all_mean)).to eq(copied.instance_variable_get(:@all_mean))
      expect(((z_test - copied.transform(kernel_mat_test))**2).sum).to be < 1.0e-8
    end

    context 'when one-dimensional subspace' do
      let(:n_components) { 1 }
      let(:pca) { Rumale::Decomposition::PCA.new(n_components: n_components, solver: 'evd').fit(x_train) }
      let(:z_train_pca) { pca.transform(x_train) }
      let(:z_test_pca) { pca.transform(x_test) }

      it 'maps data into one-dimensional subspace.', :aggregate_failures do
        expect(z_train.class).to eq(Numo::DFloat)
        expect(z_train.shape[0]).to eq(n_train_samples)
        expect(z_train.shape[1]).to eq(nil)
        expect(z_test.class).to eq(Numo::DFloat)
        expect(z_test.shape[0]).to eq(n_test_samples)
        expect(z_test.shape[1]).to eq(nil)
      end

      it 'has small error from PCA in the case of a linear kernel.', :aggregate_failures do
        expect(Numo::NMath.sqrt(((z_train - z_train_pca)**2).sum)).to be < 1e-8
        expect(Numo::NMath.sqrt(((z_test - z_test_pca)**2).sum)).to be < 1e-8
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
    let(:classifier) { Rumale::LinearModel::SVC.new(reg_param: 0.01, fit_bias: true, random_seed: 1) }

    before { classifier.fit(z_train, y_train) }

    it 'maps to a linearly separable space', :aggregate_failures do
      expect(z_train.class).to eq(Numo::DFloat)
      expect(z_train.shape[0]).to eq(n_train_samples)
      expect(z_train.shape[1]).to eq(n_components)
      expect(z_test.class).to eq(Numo::DFloat)
      expect(z_test.shape[0]).to eq(n_test_samples)
      expect(z_test.shape[1]).to eq(n_components)
      expect(classifier.score(z_train, y_train)).to be_within(0.01).of(1.0)
      expect(classifier.score(z_test, y_test)).to be_within(0.01).of(1.0)
    end
  end
end
