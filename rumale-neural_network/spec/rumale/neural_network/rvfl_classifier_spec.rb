# frozen_string_literal: true

require 'spec_helper'
require 'numo/linalg'

RSpec.describe Rumale::NeuralNetwork::RVFLClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:hidden_units) { 64 }
  let(:estimator) { described_class.new(hidden_units: hidden_units, reg_param: 1e4, random_seed: 1) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  shared_examples 'classification' do
    before { estimator.fit(x, y) }

    it 'classifies given dataset.', :aggregate_failures do
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.random_weight_vec).to be_a(Numo::DFloat)
      expect(estimator.random_weight_vec).to be_contiguous
      expect(estimator.random_weight_vec.ndim).to eq(2)
      expect(estimator.random_weight_vec.shape[0]).to eq(n_features)
      expect(estimator.random_weight_vec.shape[1]).to eq(hidden_units)
      expect(estimator.random_bias).to be_a(Numo::DFloat)
      expect(estimator.random_bias).to be_contiguous
      expect(estimator.random_bias.ndim).to eq(1)
      expect(estimator.random_bias.shape[0]).to eq(hidden_units)
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_features + hidden_units)
      expect(estimator.weight_vec.shape[1]).to eq(n_classes)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(score).to eq(1.0)
    end
  end

  context 'when the number of hidden units is less than the number of samples' do
    context 'when binary classification problem' do
      let(:dataset) { xor_dataset }

      it_behaves_like 'classification'
    end

    context 'when multiclass classification problem' do
      let(:dataset) { three_clusters_dataset }

      it_behaves_like 'classification'
    end
  end

  context 'when the number of hidden units is greater than the number of samples' do
    let(:hidden_units) { 512 }

    context 'when binary classification problem' do
      let(:dataset) { xor_dataset }

      it_behaves_like 'classification'
    end

    context 'when multiclass classification problem' do
      let(:dataset) { three_clusters_dataset }

      it_behaves_like 'classification'
    end
  end
end
