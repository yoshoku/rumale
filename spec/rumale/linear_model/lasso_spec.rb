# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Lasso do
  let(:x) { two_clusters_dataset[0] }
  let(:single_target) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:single_target_with_bias) { 0.1 + single_target }
  let(:multi_target) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { multi_target.shape[1] }
  let(:fit_bias) { false }
  let(:n_jobs) { nil }
  let(:estimator) { described_class.new(reg_param: 0.1, fit_bias: fit_bias, n_jobs: n_jobs, random_seed: 1).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when single target problem' do
    let(:y) { single_target }

    it 'learns the model for single target problem.', :aggregate_failures do
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.bias_term).to be_zero
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
      expect(estimator.params[:fit_bias]).to eq(copied.params[:fit_bias])
      expect(estimator.params[:bias_scale]).to eq(copied.params[:bias_scale])
      expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
      expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
      expect(estimator.params[:optimizer].class).to eq(copied.params[:optimizer].class)
      expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
      expect(estimator.weight_vec).to eq(copied.weight_vec)
      expect(estimator.bias_term).to eq(copied.bias_term)
      expect(estimator.rng).to eq(copied.rng)
      expect(score).to eq(copied.score(x, y))
    end

    context 'when add bias to target values' do
      let(:fit_bias) { true }
      let(:y) { single_target_with_bias }

      it 'learns the model for single regression problem with bias term.', :aggregate_failures do
        expect(estimator.weight_vec.ndim).to eq(1)
        expect(estimator.weight_vec.shape[0]).to eq(n_features)
        expect(estimator.bias_term).not_to be_zero
        expect(score).to be_within(0.01).of(1.0)
      end
    end
  end

  context 'when multi-target problem' do
    let(:y) { multi_target }

    it 'learns the model for multiple-regression problems.', :aggregate_failures do
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.01).of(1.0)
    end

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'learns the model for multiple-regression problems in parallel.', :aggregate_failures do
        expect(estimator.weight_vec.class).to eq(Numo::DFloat)
        expect(estimator.weight_vec.ndim).to eq(2)
        expect(estimator.weight_vec.shape[0]).to eq(n_features)
        expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
        expect(estimator.bias_term.class).to eq(Numo::DFloat)
        expect(estimator.bias_term.ndim).to eq(1)
        expect(estimator.bias_term.shape[0]).to eq(n_outputs)
        expect(predicted.class).to eq(Numo::DFloat)
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.01).of(1.0)
      end
    end
  end
end
