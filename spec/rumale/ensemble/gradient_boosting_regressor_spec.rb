# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::GradientBoostingRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_estimators) { 10 }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(n_estimators: n_estimators, learning_rate: 0.9, reg_lambda: 0.001, max_features: 1,
                        n_jobs: n_jobs, random_seed: 9).fit(x, y)
  end
  let(:leaf_ids) { estimator.apply(x) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  context 'when single target problem' do
    let(:y) { x[true, 0] + x[true, 1]**2 }
    let(:copied) { Marshal.load(Marshal.dump(estimator)) }

    it 'learns the model for single regression problem.', :aggregate_failures do
      expect(estimator.params[:n_estimators]).to eq(n_estimators)
      expect(estimator.params[:max_features]).to eq(1)
      expect(estimator.estimators.class).to eq(Array)
      expect(estimator.estimators[0].class).to eq(Rumale::Tree::GradientTreeRegressor)
      expect(estimator.estimators.size).to eq(n_estimators)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.02).of(1.0)
      expect(leaf_ids.class).to eq(Numo::Int32)
      expect(leaf_ids.ndim).to eq(2)
      expect(leaf_ids.shape[0]).to eq(n_samples)
      expect(leaf_ids.shape[1]).to eq(n_estimators)
      expect(leaf_ids[true, 0]).to eq(estimator.estimators[0].apply(x))
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params).to match(copied.params)
      expect(estimator.estimators.size).to eq(copied.estimators.size)
      expect(estimator.feature_importances).to eq(copied.feature_importances)
      expect(estimator.rng).to eq(copied.rng)
      expect(score).to eq(copied.score(x, y))
    end

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'learns the model for single regression problem in parallel.', :aggregate_failures do
        expect(estimator.params[:n_estimators]).to eq(n_estimators)
        expect(estimator.params[:max_features]).to eq(1)
        expect(estimator.estimators.class).to eq(Array)
        expect(estimator.estimators[0].class).to eq(Rumale::Tree::GradientTreeRegressor)
        expect(estimator.estimators.size).to eq(n_estimators)
        expect(estimator.feature_importances.class).to eq(Numo::DFloat)
        expect(estimator.feature_importances.ndim).to eq(1)
        expect(estimator.feature_importances.shape[0]).to eq(n_features)
        expect(predicted.class).to eq(Numo::DFloat)
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(score).to be_within(0.02).of(1.0)
        expect(leaf_ids.class).to eq(Numo::Int32)
        expect(leaf_ids.ndim).to eq(2)
        expect(leaf_ids.shape[0]).to eq(n_samples)
        expect(leaf_ids.shape[1]).to eq(n_estimators)
        expect(leaf_ids[true, 0]).to eq(estimator.estimators[0].apply(x))
      end
    end
  end

  context 'when multi-target problem' do
    let(:y) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.0, 0.1]]) }
    let(:n_outputs) { y.shape[1] }

    it 'learns the model for multiple regression problem.', :aggregate_failures do
      expect(estimator.estimators.class).to eq(Array)
      expect(estimator.estimators[0].class).to eq(Array)
      expect(estimator.estimators[0][0].class).to eq(Rumale::Tree::GradientTreeRegressor)
      expect(estimator.estimators.size).to eq(n_outputs)
      expect(estimator.estimators[0].size).to eq(n_estimators)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.02).of(1.0)
      expect(leaf_ids.class).to eq(Numo::Int32)
      expect(leaf_ids.ndim).to eq(3)
      expect(leaf_ids.shape[0]).to eq(n_samples)
      expect(leaf_ids.shape[1]).to eq(n_estimators)
      expect(leaf_ids.shape[2]).to eq(n_outputs)
    end

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'learns the model for multiple regression problem in parallel.', :aggregate_failures do
        expect(estimator.estimators.class).to eq(Array)
        expect(estimator.estimators[0].class).to eq(Array)
        expect(estimator.estimators[0][0].class).to eq(Rumale::Tree::GradientTreeRegressor)
        expect(estimator.estimators.size).to eq(n_outputs)
        expect(estimator.estimators[0].size).to eq(n_estimators)
        expect(estimator.feature_importances.class).to eq(Numo::DFloat)
        expect(estimator.feature_importances.ndim).to eq(1)
        expect(estimator.feature_importances.shape[0]).to eq(n_features)
        expect(predicted.class).to eq(Numo::DFloat)
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.02).of(1.0)
        expect(leaf_ids.class).to eq(Numo::Int32)
        expect(leaf_ids.ndim).to eq(3)
        expect(leaf_ids.shape[0]).to eq(n_samples)
        expect(leaf_ids.shape[1]).to eq(n_estimators)
        expect(leaf_ids.shape[2]).to eq(n_outputs)
      end
    end
  end
end
