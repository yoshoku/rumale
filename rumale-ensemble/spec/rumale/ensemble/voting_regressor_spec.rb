# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::VotingRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:estimators) do
    { dtr: Rumale::Tree::DecisionTreeRegressor.new(random_seed: 1),
      mlp: Rumale::NeuralNetwork::MLPRegressor.new(hidden_units: [32], max_iter: 50, random_seed: 1),
      rdg: Rumale::LinearModel::LinearRegression.new }
  end
  let(:weights) do
    { dtr: 0.6, mlp: 0.3, rdg: 0.1 }
  end
  let(:estimator) { described_class.new(estimators: estimators, weights: weights).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  context 'when single target problem' do
    let(:y) { x[true, 0] + x[true, 1]**2 }

    it 'learns the model for single regression problem', :aggregate_failures do
      expect(estimator.estimators).to be_a(Hash)
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end
  end

  context 'when multi-target problem' do
    let(:y) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.0, 0.1]]) }
    let(:n_outputs) { y.shape[1] }

    it 'learns the model for multiple regression problem', :aggregate_failures do
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end
  end
end
