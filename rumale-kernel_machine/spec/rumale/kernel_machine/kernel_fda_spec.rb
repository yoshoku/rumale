# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::KernelMachine::KernelFDA do
  let(:n_samples) { x.shape[0] }
  let(:n_components) { nil }
  let(:transformer) { described_class.new(n_components: n_components) }
  let(:dataset_ids) { Array(0...n_samples) }
  let(:train_ids) { dataset_ids.sample(n_samples * 0.9, random: Random.new(1)) }
  let(:test_ids) { dataset_ids - train_ids }
  let(:x_train) { x[train_ids, true].dup }
  let(:x_test) { x[test_ids, true].dup }
  let(:y_train) { y[train_ids].dup }
  let(:y_test) { y[test_ids].dup }
  let(:n_train_samples) { x_train.shape[0] }
  let(:n_test_samples) { x_test.shape[0] }

  describe 'basic examples' do
    let(:dataset) { three_clusters_dataset }
    let(:x) { dataset[0] }
    let(:y) { dataset[1] }
    let(:n_classes) { y.to_a.uniq.size - 1 }
    let(:kernel_mat_train) { Rumale::PairwiseMetric.linear_kernel(x_train, nil) }
    let(:kernel_mat_test) { Rumale::PairwiseMetric.linear_kernel(x_test, x_train) }
    let(:z_train) { transformer.fit_transform(kernel_mat_train, y_train) }
    let(:z_test) { transformer.transform(kernel_mat_test) }

    it 'maps into subspace', :aggregate_failures do
      expect(z_train).to be_a(Numo::DFloat)
      expect(z_train).to be_contiguous
      expect(z_train.ndim).to eq(2)
      expect(z_train.shape[0]).to eq(n_train_samples)
      expect(z_train.shape[1]).to eq(n_classes)
      expect(z_test).to be_a(Numo::DFloat)
      expect(z_test).to be_contiguous
      expect(z_test.ndim).to eq(2)
      expect(z_test.shape[0]).to eq(n_test_samples)
      expect(z_test.shape[1]).to eq(n_classes)
      expect(transformer.alphas).to be_a(Numo::DFloat)
      expect(transformer.alphas).to be_contiguous
      expect(transformer.alphas.ndim).to eq(2)
      expect(transformer.alphas.shape[0]).to eq(n_train_samples)
      expect(transformer.alphas.shape[1]).to eq(n_classes)
    end
  end

  describe 'using with nearest neighbor classifier' do
    let(:dataset) { Rumale::Dataset.make_circles(200, factor: 0.4, noise: 0.03, random_seed: 1) }
    let(:x) { dataset[0] }
    let(:y) { dataset[1] }
    let(:n_components) { 1 }
    let(:kernel_mat_train) { Rumale::PairwiseMetric.rbf_kernel(x_train, nil, 1.0) }
    let(:kernel_mat_test) { Rumale::PairwiseMetric.rbf_kernel(x_test, x_train, 1.0) }
    let(:z_train) { transformer.fit_transform(kernel_mat_train, y_train) }
    let(:z_test) { transformer.transform(kernel_mat_test) }
    let(:classifier) { Rumale::NearestNeighbors::KNeighborsClassifier.new(n_neighbors: 1).fit(z_train.expand_dims(1), y_train) }
    let(:train_score) { classifier.score(z_train.expand_dims(1), y_train) }
    let(:test_score) { classifier.score(z_test.expand_dims(1), y_test) }

    it 'maps to a linearly separable space', :aggregate_failures do
      expect(z_train).to be_a(Numo::DFloat)
      expect(z_train).to be_contiguous
      expect(z_train.ndim).to eq(1)
      expect(z_train.shape[0]).to eq(n_train_samples)
      expect(z_test).to be_a(Numo::DFloat)
      expect(z_test).to be_contiguous
      expect(z_test.ndim).to eq(1)
      expect(z_test.shape[0]).to eq(n_test_samples)
      expect(train_score).to be_within(0.01).of(1.0)
      expect(test_score).to be_within(0.01).of(1.0)
    end
  end
end
