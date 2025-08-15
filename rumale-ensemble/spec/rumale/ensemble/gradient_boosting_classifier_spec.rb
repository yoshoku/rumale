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
    described_class.new(n_estimators: n_estimators, learning_rate: 0.9, max_features: 1, n_jobs: n_jobs, random_seed: 1).fit(x,
                                                                                                                             y)
  end
  let(:score) { estimator.score(x, y) }
  let(:leaf_ids) { estimator.apply(x) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }

    it 'classifies two clusters data', :aggregate_failures do
      expect(estimator.estimators).to be_a(Array)
      expect(estimator.estimators[0]).to be_a(Rumale::Tree::GradientTreeRegressor)
      expect(estimator.estimators.size).to eq(n_estimators)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances).to be_a(Numo::DFloat)
      expect(estimator.feature_importances).to be_contiguous
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(score).to be_within(0.02).of(1.0)
      expect(leaf_ids).to be_a(Numo::Int32)
      expect(leaf_ids).to be_contiguous
      expect(leaf_ids.ndim).to eq(2)
      expect(leaf_ids.shape[0]).to eq(n_samples)
      expect(leaf_ids.shape[1]).to eq(n_estimators)
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }
    let(:probs) { estimator.predict_proba(x) }
    let(:predicted_by_probs) { Numo::Int32[*Array.new(n_samples) { |n| classes[probs[n, true].max_index] }] }

    it 'classifies three clusters data', :aggregate_failures do
      expect(estimator.estimators).to be_a(Array)
      expect(estimator.estimators[0]).to be_a(Array)
      expect(estimator.estimators[0][0]).to be_a(Rumale::Tree::GradientTreeRegressor)
      expect(estimator.estimators.size).to eq(3)
      expect(estimator.estimators[0].size).to eq(n_estimators)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances).to be_a(Numo::DFloat)
      expect(estimator.feature_importances).to be_contiguous
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(score).to be_within(0.02).of(1.0)
      expect(leaf_ids).to be_a(Numo::Int32)
      expect(leaf_ids).to be_contiguous
      expect(leaf_ids.ndim).to eq(3)
      expect(leaf_ids.shape[0]).to eq(n_samples)
      expect(leaf_ids.shape[1]).to eq(n_estimators)
      expect(leaf_ids.shape[2]).to eq(3)
    end

    it 'estimates class probabilities with three clusters dataset', :aggregate_failures do
      expect(probs).to be_a(Numo::DFloat)
      expect(probs).to be_contiguous
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predicted_by_probs).to eq(y)
    end

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'classifies three clusters data in parallel', :aggregate_failures do
        expect(estimator.estimators).to be_a(Array)
        expect(estimator.estimators[0]).to be_a(Array)
        expect(estimator.estimators[0][0]).to be_a(Rumale::Tree::GradientTreeRegressor)
        expect(estimator.estimators.size).to eq(3)
        expect(estimator.estimators[0].size).to eq(n_estimators)
        expect(estimator.classes).to be_a(Numo::Int32)
        expect(estimator.classes).to be_contiguous
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(estimator.feature_importances).to be_a(Numo::DFloat)
        expect(estimator.feature_importances).to be_contiguous
        expect(estimator.feature_importances.ndim).to eq(1)
        expect(estimator.feature_importances.shape[0]).to eq(n_features)
        expect(score).to be_within(0.02).of(1.0)
        expect(leaf_ids).to be_a(Numo::Int32)
        expect(leaf_ids).to be_contiguous
        expect(leaf_ids.ndim).to eq(3)
        expect(leaf_ids.shape[0]).to eq(n_samples)
        expect(leaf_ids.shape[1]).to eq(n_estimators)
        expect(leaf_ids.shape[2]).to eq(3)
      end
    end
  end
end
