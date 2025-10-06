# frozen_string_literal: true

require 'spec_helper'
require 'numo/linalg'

RSpec.describe Rumale::NeuralNetwork::RBFRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:single_target) { x[true, 0] + x[true, 1]**2 }
  let(:multi_target) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.8, 0.2]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { y.shape[1] }
  let(:hidden_units) { 64 }
  let(:estimator) { described_class.new(hidden_units: hidden_units, reg_param: 1e4, random_seed: 1) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  shared_examples 'regression' do
    before { estimator.fit(x, y) }

    it 'fits model for given dataset.', :aggregate_failures do
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(y.ndim)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(estimator.centers).to be_a(Numo::DFloat)
      expect(estimator.centers).to be_contiguous
      expect(estimator.centers.ndim).to eq(2)
      expect(estimator.centers.shape[0]).to eq(hidden_units)
      expect(estimator.centers.shape[1]).to eq(n_features)
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(hidden_units)
      expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
      expect(score).to be > 0.98
    end
  end

  context 'when the number of hidden units is less than the number of samples' do
    context 'when single regression problem' do
      let(:y) { single_target }
      let(:n_outputs) { 1 }

      it_behaves_like 'regression'
    end

    context 'when multiple regression problem' do
      let(:y) { multi_target }

      it_behaves_like 'regression'
    end
  end

  context 'when the number of hidden units is greater than the number of samples' do
    let(:hidden_units) { 400 }

    context 'when single regression problem' do
      let(:y) { single_target }
      let(:n_outputs) { 1 }

      it_behaves_like 'regression'
    end

    context 'when multiple regression problem' do
      let(:y) { multi_target }

      it_behaves_like 'regression'
    end
  end
end
