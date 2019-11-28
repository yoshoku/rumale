# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::GradientBoostingClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:n_estimators) { 10 }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(n_estimators: n_estimators, learning_rate: 0.9, max_features: 1, n_jobs: n_jobs, random_seed: 1).fit(x, y)
  end
  let(:score) { estimator.score(x, y) }
  let(:leaf_ids) { estimator.apply(x) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }

    it 'classifies two clusters data.', :aggregate_failures do
      expect(estimator.estimators.class).to eq(Array)
      expect(estimator.estimators[0].class).to eq(Rumale::Tree::GradientTreeRegressor)
      expect(estimator.estimators.size).to eq(n_estimators)
      expect(estimator.classes.class).to eq(Numo::Int32)
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(score).to be_within(0.02).of(1.0)
      expect(leaf_ids.class).to eq(Numo::Int32)
      expect(leaf_ids.ndim).to eq(2)
      expect(leaf_ids.shape[0]).to eq(n_samples)
      expect(leaf_ids.shape[1]).to eq(n_estimators)
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }
    let(:probs) { estimator.predict_proba(x) }
    let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
    let(:copied) { Marshal.load(Marshal.dump(estimator)) }

    it 'classifies three clusters data.', :aggregate_failures do
      expect(estimator.estimators.class).to eq(Array)
      expect(estimator.estimators[0].class).to eq(Array)
      expect(estimator.estimators[0][0].class).to eq(Rumale::Tree::GradientTreeRegressor)
      expect(estimator.estimators.size).to eq(3)
      expect(estimator.estimators[0].size).to eq(n_estimators)
      expect(estimator.classes.class).to eq(Numo::Int32)
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances.class).to eq(Numo::DFloat)
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(score).to be_within(0.02).of(1.0)
      expect(leaf_ids.class).to eq(Numo::Int32)
      expect(leaf_ids.ndim).to eq(3)
      expect(leaf_ids.shape[0]).to eq(n_samples)
      expect(leaf_ids.shape[1]).to eq(n_estimators)
      expect(leaf_ids.shape[2]).to eq(3)
    end

    it 'estimates class probabilities with three clusters dataset.', :aggregate_failures do
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predicted_by_probs).to eq(y)
    end

    it 'dumps and restores itself using Marshal module.', :aggeregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params).to match(copied.params)
      expect(estimator.estimators.size).to eq(copied.estimators.size)
      expect(estimator.classes).to eq(copied.classes)
      expect(estimator.feature_importances).to eq(copied.feature_importances)
      expect(estimator.rng).to eq(copied.rng)
      expect(score).to eq(copied.score(x, y))
    end

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'classifies three clusters data in parallel.', :aggregate_failures do
        expect(estimator.estimators.class).to eq(Array)
        expect(estimator.estimators[0].class).to eq(Array)
        expect(estimator.estimators[0][0].class).to eq(Rumale::Tree::GradientTreeRegressor)
        expect(estimator.estimators.size).to eq(3)
        expect(estimator.estimators[0].size).to eq(n_estimators)
        expect(estimator.classes.class).to eq(Numo::Int32)
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(estimator.feature_importances.class).to eq(Numo::DFloat)
        expect(estimator.feature_importances.ndim).to eq(1)
        expect(estimator.feature_importances.shape[0]).to eq(n_features)
        expect(score).to be_within(0.02).of(1.0)
        expect(leaf_ids.class).to eq(Numo::Int32)
        expect(leaf_ids.ndim).to eq(3)
        expect(leaf_ids.shape[0]).to eq(n_samples)
        expect(leaf_ids.shape[1]).to eq(n_estimators)
        expect(leaf_ids.shape[2]).to eq(3)
      end
    end
  end
end
