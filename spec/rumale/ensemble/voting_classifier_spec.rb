# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::VotingClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_classes) { classes.size }
  let(:estimators) do
    { dtr: Rumale::Tree::DecisionTreeClassifier.new,
      gnb: Rumale::NaiveBayes::GaussianNB.new,
      lgr: Rumale::LinearModel::LogisticRegression.new }
  end
  let(:weights) { nil }
  let(:voting) { 'hard' }
  let(:estimator) { described_class.new(estimators: estimators, weights: weights, voting: voting).fit(x, y) }
  let(:func_vals) { estimator.decision_function(x) }
  let(:probs) { estimator.predict_proba(x) }
  let(:predicted) { estimator.predict(x) }
  let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
  let(:score) { estimator.score(x, y) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }
    let(:copied) { Marshal.load(Marshal.dump(estimator)) }

    it 'classifies two clusters data.', :aggregate_failures do
      expect(estimator.params[:weights]).to match({ dtr: 1, gnb: 1, lgr: 1 })
      expect(estimator.params[:voting]).to eq('hard')
      expect(estimator.estimators).to be_a(Hash)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(func_vals).to be_a(Numo::DFloat)
      expect(func_vals).to be_contiguous
      expect(func_vals.ndim).to eq(2)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predicted_by_probs).to eq(y)
      expect(score).to eq(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(copied.class).to eq(estimator.class)
      expect(copied.params).to match(estimator.params)
      expect(copied.estimators.keys).to eq(estimator.estimators.keys)
      expect(copied.classes).to eq(estimator.classes)
      expect(copied.score(x, y)).to eq(score)
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }

    it 'classifies three clusters data.', :aggregate_failures do
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
      expect(predicted_by_probs).to eq(y)
      expect(score).to eq(1.0)
    end
  end

  context "when use 'soft' voting" do
    let(:dataset) { xor_dataset }
    let(:voting) { 'soft' }
    let(:weights) do
      { dtr: 0.7, gnb: 0.1, lgr: 0.2 }
    end

    it 'classifies xor dataset.', :aggregate_failures do
      expect(estimator.params[:weights]).to match({ dtr: 0.7, gnb: 0.1, lgr: 0.2 })
      expect(estimator.params[:voting]).to eq('soft')
      expect(estimator.estimators).to be_a(Hash)
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(func_vals).to be_a(Numo::DFloat)
      expect(func_vals).to be_contiguous
      expect(func_vals.ndim).to eq(2)
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
      expect(predicted_by_probs).to eq(y)
      expect(score).to eq(1.0)
    end
  end
end
