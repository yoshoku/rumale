# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::AdaBoostRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:estimator) { described_class.new(n_estimators: 10, criterion: 'mae', max_features: 2, random_seed: 9).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  context 'when single target problem' do
    let(:y) { x[true, 0] + x[true, 1]**2 }
    let(:copied) { Marshal.load(Marshal.dump(estimator)) }

    it 'learns the model for single regression problem.', :aggregate_failures do
      expect(estimator.estimators.class).to eq(Array)
      expect(estimator.estimators[0].class).to eq(Rumale::Tree::DecisionTreeRegressor)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(estimator.estimator_weights.class).to eq(Numo::DFloat)
      expect(estimator.estimator_weights.ndim).to eq(1)
      expect(estimator.estimator_weights.shape[0]).to eq(estimator.estimators.size)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.estimators.size).to eq(copied.estimators.size)
      expect(estimator.estimator_weights).to eq(copied.estimator_weights)
      expect(estimator.feature_importances).to eq(copied.feature_importances)
      expect(estimator.rng).to eq(copied.rng)
      expect(score).to eq(copied.score(x, y))
    end
  end

  context 'when multi-target problem' do
    let(:y) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.0, 0.1]]) }

    it 'raises ArgumentError when given multiple target values.' do
      expect { estimator.fit(x, y) }.to raise_error(ArgumentError)
    end
  end
end
