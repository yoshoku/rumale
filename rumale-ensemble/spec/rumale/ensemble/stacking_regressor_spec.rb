# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::StackingRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:estimators) do
    { dtr: Rumale::Tree::DecisionTreeRegressor.new(random_seed: 1),
      mlp: Rumale::NeuralNetwork::MLPRegressor.new(hidden_units: [8], max_iter: 50, random_seed: 1),
      rdg: Rumale::LinearModel::LinearRegression.new }
  end
  let(:n_base_estimators) { estimators.size }
  let(:meta_estimator) { nil }
  let(:passthrough) { false }
  let(:estimator) do
    described_class.new(estimators: estimators, meta_estimator: meta_estimator, passthrough: passthrough, random_seed: 1)
  end
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  context 'when single target problem' do
    let(:y) { x[true, 0] + x[true, 1]**2 }

    before { estimator.fit(x, y) }

    it 'learns the model for single regression problem', :aggregate_failures do
      expect(estimator.params[:n_splits]).to eq(5)
      expect(estimator.params[:shuffle]).to be_truthy
      expect(estimator.params[:passthrough]).to be_falsy
      expect(estimator.estimators).to be_a(Hash)
      expect(estimator.meta_estimator).to be_a(Rumale::LinearModel::Ridge)
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
    let(:meta_estimator) { Rumale::LinearModel::Lasso.new }

    before { estimator.fit(x, y) }

    it 'learns the model for multiple regression problem', :aggregate_failures do
      expect(estimator.meta_estimator).to be_a(Rumale::LinearModel::Lasso)
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    context 'when used as feature extractor' do
      let(:meta_features) { estimator.fit_transform(x, y) }
      let(:n_components) { n_outputs * n_base_estimators }

      it 'extracts meta features', :aggregate_failures do
        expect(meta_features).to be_a(Numo::DFloat)
        expect(meta_features).to be_contiguous
        expect(meta_features.ndim).to eq(2)
        expect(meta_features.shape[0]).to eq(n_samples)
        expect(meta_features.shape[1]).to eq(n_components)
      end

      context 'when concatenating original features' do
        let(:passthrough) { true }

        it 'extracts meta features concatenated with original features', :aggregate_failures do
          expect(meta_features).to be_a(Numo::DFloat)
          expect(meta_features).to be_contiguous
          expect(meta_features.ndim).to eq(2)
          expect(meta_features.shape[0]).to eq(n_samples)
          expect(meta_features.shape[1]).to eq(n_components + n_features)
        end
      end
    end
  end
end
