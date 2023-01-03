# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::SGDRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:y) { single_target }
  let(:single_target) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:multi_target) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { multi_target.shape[1] }
  let(:fit_bias) { false }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(loss: loss, reg_param: 1, epsilon: 0.1, fit_bias: fit_bias, n_jobs: n_jobs, random_seed: 1).fit(x, y)
  end
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  shared_examples 'regression problems' do
    context 'when single target problem' do
      let(:y) { single_target }

      it 'learns the linear model.', :aggregate_failures do
        expect(estimator.weight_vec).to be_a(Numo::DFloat)
        expect(estimator.weight_vec).to be_contiguous
        expect(estimator.weight_vec.ndim).to eq(1)
        expect(estimator.weight_vec.shape[0]).to eq(n_features)
        expect(estimator.bias_term).to be_zero
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(score).to be_within(0.01).of(1.0)
      end

      context 'when fit_bias parameter is true' do
        let(:fit_bias) { true }

        it 'learns the linear model with bias term.', :aggregate_failures do
          expect(estimator.weight_vec.ndim).to eq(1)
          expect(estimator.weight_vec.shape[0]).to eq(n_features)
          expect(estimator.bias_term).not_to be_zero
          expect(score).to be_within(0.01).of(1.0)
        end
      end
    end

    context 'when multi-target problem' do
      let(:y) { multi_target }

      it 'learns the model for multi-target problem.', :aggregate_failures do
        expect(estimator.weight_vec).to be_a(Numo::DFloat)
        expect(estimator.weight_vec).to be_contiguous
        expect(estimator.weight_vec.ndim).to eq(2)
        expect(estimator.weight_vec.shape[0]).to eq(n_features)
        expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.01).of(1.0)
      end

      context 'when n_jobs parameter is not nil' do
        let(:n_jobs) { -1 }

        it 'learns the model for multiple-regression problems in parallel.', :aggregate_failures do
          expect(estimator.weight_vec).to be_a(Numo::DFloat)
          expect(estimator.weight_vec).to be_contiguous
          expect(estimator.weight_vec.ndim).to eq(2)
          expect(estimator.weight_vec.shape[0]).to eq(n_features)
          expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
          expect(estimator.bias_term).to be_a(Numo::DFloat)
          expect(estimator.bias_term).to be_contiguous
          expect(estimator.bias_term.ndim).to eq(1)
          expect(estimator.bias_term.shape[0]).to eq(n_outputs)
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

  context 'when loss is "squared_error"' do
    let(:loss) { 'squared_error' }

    it_behaves_like 'regression problems'

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(copied.class).to eq(estimator.class)
      expect(copied.params).to eq(estimator.params)
      expect(copied.weight_vec).to eq(estimator.weight_vec)
      expect(copied.bias_term).to eq(estimator.bias_term)
      expect(copied.rng).to eq(estimator.rng)
      expect(copied.score(x, y)).to eq(score)
      expect(copied.instance_variable_get(:@penalty_type)).to eq('l2')
      expect(copied.instance_variable_get(:@loss_func)).to be_a(Rumale::LinearModel::Loss::MeanSquaredError)
    end
  end

  context 'when loss is "epsilon_insensitive"' do
    let(:loss) { 'epsilon_insensitive' }

    it_behaves_like 'regression problems'

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(copied.class).to eq(estimator.class)
      expect(copied.params).to eq(estimator.params)
      expect(copied.weight_vec).to eq(estimator.weight_vec)
      expect(copied.bias_term).to eq(estimator.bias_term)
      expect(copied.rng).to eq(estimator.rng)
      expect(copied.score(x, y)).to eq(score)
      expect(copied.instance_variable_get(:@penalty_type)).to eq('l2')
      expect(copied.instance_variable_get(:@loss_func)).to be_a(Rumale::LinearModel::Loss::EpsilonInsensitive)
    end
  end
end
