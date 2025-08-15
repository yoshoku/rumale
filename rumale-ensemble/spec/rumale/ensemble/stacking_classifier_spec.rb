# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::StackingClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:estimators) do
    { dtr: Rumale::Tree::DecisionTreeClassifier.new,
      gnb: Rumale::NaiveBayes::GaussianNB.new,
      svc: Rumale::LinearModel::LogisticRegression.new }
  end
  let(:n_base_estimators) { estimators.size }
  let(:meta_estimator) { nil }
  let(:stack_method) { 'auto' }
  let(:passthrough) { false }
  let(:estimator) do
    described_class.new(
      estimators: estimators, meta_estimator: meta_estimator, stack_method: stack_method, passthrough: passthrough,
      random_seed: 1
    )
  end
  let(:func_vals) { estimator.decision_function(x) }
  let(:predicted) { estimator.predict(x) }
  let(:probs) { estimator.predict_proba(x) }
  let(:score) { estimator.score(x, y) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }

    before { estimator.fit(x, y) }

    it 'classifies two clusters data', :aggregate_failures do
      expect(estimator.params[:n_splits]).to eq(5)
      expect(estimator.params[:shuffle]).to be_truthy
      expect(estimator.params[:stack_method]).to eq('auto')
      expect(estimator.params[:passthrough]).to be_falsy
      expect(estimator.estimators).to be_a(Hash)
      expect(estimator.meta_estimator).to be_a(Rumale::LinearModel::LogisticRegression)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.stack_method).to eq({ dtr: :predict_proba, gnb: :predict_proba, svc: :predict_proba })
      expect(func_vals).to be_a(Numo::DFloat)
      expect(func_vals).to be_contiguous
      expect(func_vals.ndim).to eq(1)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(probs).to be_a(Numo::DFloat)
      expect(probs).to be_contiguous
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(score).to eq(1.0)
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }
    let(:meta_estimator) { Rumale::LinearModel::SVC.new(probability: true) }
    let(:predicted_by_probs) { Numo::Int32[*Array.new(n_samples) { |n| classes[probs[n, true].max_index] }] }

    before { estimator.fit(x, y) }

    it 'classifies three clusters data', :aggregate_failures do
      expect(estimator.meta_estimator).to be_a(Rumale::LinearModel::SVC)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(func_vals).to be_a(Numo::DFloat)
      expect(func_vals).to be_contiguous
      expect(func_vals.ndim).to eq(2)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(func_vals.shape[1]).to eq(n_classes)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(probs).to be_a(Numo::DFloat)
      expect(probs).to be_contiguous
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(score).to eq(1.0)
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

  context 'when used as feature extractor' do
    let(:dataset) { three_clusters_dataset }
    let(:meta_features) { estimator.fit_transform(x, y) }

    it 'extracts meta features', :aggregate_failures do
      expect(meta_features).to be_a(Numo::DFloat)
      expect(meta_features).to be_contiguous
      expect(meta_features.ndim).to eq(2)
      expect(meta_features.shape[0]).to eq(n_samples)
      expect(meta_features.shape[1]).to eq(n_classes * n_base_estimators)
    end

    context 'when concatenating original features' do
      let(:passthrough) { true }
      let(:stack_method) { 'predict' }

      it 'extracts meta features concatenated with original features', :aggregate_failures do
        expect(meta_features).to be_a(Numo::DFloat)
        expect(meta_features).to be_contiguous
        expect(meta_features.ndim).to eq(2)
        expect(meta_features.shape[0]).to eq(n_samples)
        expect(meta_features.shape[1]).to eq(n_classes + n_features)
      end
    end
  end
end
