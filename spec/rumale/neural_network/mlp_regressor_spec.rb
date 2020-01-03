# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NeuralNetwork::MLPRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:single_target) { x[true, 0] + x[true, 1]**2 }
  let(:multi_target) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.8, 0.2]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { y.shape[1] }
  let(:estimator) { described_class.new(hidden_units: [128, 128], max_iter: 100, verbose: false, random_seed: 1).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  shared_examples 'regression' do
    it 'fits model for given dataset.', :aggregate_failures do
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(y.ndim)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be > 0.98
    end
  end

  shared_examples 'dump and load model' do
    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params).to eq(copied.params)
      expect(score).to eq(copied.score(x, y))
    end
  end

  context 'when single regression problem' do
    let(:y) { single_target }

    it_behaves_like 'regression'
    it_behaves_like 'dump and load model'
  end

  context 'when multiple regression problem' do
    let(:y) { multi_target }

    it_behaves_like 'regression'
    it_behaves_like 'dump and load model'
  end
end
