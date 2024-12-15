# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Ensemble::VRTreesRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_estimators) { 10 }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(n_estimators: n_estimators, criterion: 'mae', max_features: 2, n_jobs: n_jobs, random_seed: 9).fit(x, y)
  end
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }

  context 'when single target problem' do
    let(:y) { x[true, 0] + x[true, 1]**2 }
    let(:index_mat) { estimator.apply(x) }

    it 'learns the model for single regression problem', :aggregate_failures do
      expect(estimator.params[:n_estimators]).to eq(n_estimators)
      expect(estimator.params[:criterion]).to eq('mae')
      expect(estimator.params[:max_features]).to eq(2)
      expect(estimator.estimators).to be_a(Array)
      expect(estimator.estimators.size).to eq(n_estimators)
      expect(estimator.estimators[0]).to be_a(Rumale::Tree::VRTreeRegressor)
      expect(estimator.feature_importances).to be_a(Numo::DFloat)
      expect(estimator.feature_importances).to be_contiguous
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    it 'returns leaf index that each sample reached', :aggregate_failures do
      expect(index_mat).to be_a(Numo::Int32)
      expect(index_mat).to be_contiguous
      expect(index_mat.ndim).to eq(2)
      expect(index_mat.shape[0]).to eq(n_samples)
      expect(index_mat.shape[1]).to eq(n_estimators)
      expect(index_mat[true, 0]).to eq(estimator.estimators[0].apply(x))
    end
  end

  context 'when multi-target problem' do
    let(:y) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.0, 0.1]]) }
    let(:n_outputs) { y.shape[1] }

    it 'learns the model for multiple regression problem', :aggregate_failures do
      expect(estimator.estimators).to be_a(Array)
      expect(estimator.estimators.size).to eq(n_estimators)
      expect(estimator.estimators[0]).to be_a(Rumale::Tree::VRTreeRegressor)
      expect(estimator.feature_importances).to be_a(Numo::DFloat)
      expect(estimator.feature_importances).to be_contiguous
      expect(estimator.feature_importances.ndim).to eq(1)
      expect(estimator.feature_importances.shape[0]).to eq(n_features)
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.01).of(1.0)
    end

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'learns the model for multiple regression problem in parallel', :aggregate_failures do
        expect(estimator.estimators).to be_a(Array)
        expect(estimator.estimators.size).to eq(n_estimators)
        expect(estimator.estimators[0]).to be_a(Rumale::Tree::VRTreeRegressor)
        expect(estimator.feature_importances).to be_a(Numo::DFloat)
        expect(estimator.feature_importances).to be_contiguous
        expect(estimator.feature_importances.ndim).to eq(1)
        expect(estimator.feature_importances.shape[0]).to eq(n_features)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.01).of(1.0)
      end
    end
  end
end
