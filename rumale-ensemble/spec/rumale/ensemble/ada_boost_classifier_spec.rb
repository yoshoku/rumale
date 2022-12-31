# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::AdaBoostClassifier do
  let(:two_clusters) { two_clusters_dataset }
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_classes) { classes.size }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:estimator) { described_class.new(n_estimators: 10, max_features: 1, random_seed: 1).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:probs) { estimator.predict_proba(x) }
  let(:predicted_by_probs) { Numo::Int32.asarray(Array.new(n_samples) { |n| classes[probs[n, true].max_index] }) }
  let(:score) { estimator.score(x, y) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }

    it 'classifies two clusters data', :aggregate_failures do
      expect(estimator.estimators).to be_a(Array)
      expect(estimator.estimators[0]).to be_a(Rumale::Tree::DecisionTreeClassifier)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances).to be_a(Numo::DFloat)
      expect(estimator.feature_importances).to be_contiguous
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.03).of(1.0)
    end

    it 'estimates class probabilities with three clusters dataset', :aggregate_failures do
      expect(probs).to be_a(Numo::DFloat)
      expect(probs).to be_contiguous
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }

    it 'classifies three clusters data', :aggregate_failures do
      expect(estimator.estimators).to be_a(Array)
      expect(estimator.estimators[0]).to be_a(Rumale::Tree::DecisionTreeClassifier)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.feature_importances).to be_a(Numo::DFloat)
      expect(estimator.feature_importances).to be_contiguous
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.03).of(1.0)
    end

    it 'estimates class probabilities with three clusters dataset', :aggregate_failures do
      expect(probs).to be_a(Numo::DFloat)
      expect(probs).to be_contiguous
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predicted_by_probs).to eq(y)
    end
  end
end
