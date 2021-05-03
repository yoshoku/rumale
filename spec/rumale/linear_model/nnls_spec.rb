# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::NNLS do
  let(:x) { two_clusters_dataset[0] }
  let(:single_target) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:multi_target) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { multi_target.shape[1] }
  let(:fit_bias) { false }
  let(:estimator) { described_class.new(reg_param: 1e-4, fit_bias: fit_bias, random_seed: 1).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when single target regression problem' do
    let(:y) { single_target }

    it 'learns the model for single regression problem.', :aggregate_failures do
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.lt(0).count).to be_zero
      expect(estimator.bias_term).to be_zero
      expect(estimator.n_iter).not_to be_zero
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.class).to eq(estimator.class)
      expect(copied.params).to eq(estimator.params)
      expect(copied.weight_vec).to eq(estimator.weight_vec)
      expect(copied.bias_term).to eq(estimator.bias_term)
      expect(copied.n_iter).to eq(estimator.n_iter)
      expect(copied.rng).to eq(estimator.rng)
      expect(copied.score(x, y)).to eq(score)
    end
  end

  context 'when single regression probem with bias' do
    let(:y) { single_target + 2 }
    let(:fit_bias) { true }

    it 'learns the model for single regression problem with bias term', :aggregate_failures do
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.lt(0).count).to be_zero
      expect(estimator.bias_term).not_to be_zero
      expect(estimator.bias_term).to be_positive
      expect(score).to be_within(0.01).of(1.0)
    end
  end

  context 'when multiple regression problem' do
    let(:y) { multi_target }

    it 'learns the model for multiple-regression problems', :aggregate_failures do
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
      expect(estimator.weight_vec.lt(0).count).to be_zero
      expect(estimator.bias_term).to be_a(Numo::DFloat)
      expect(estimator.bias_term).to be_contiguous
      expect(estimator.bias_term.ndim).to eq(1)
      expect(estimator.bias_term.shape[0]).to eq(n_outputs)
      expect(estimator.bias_term.eq(0).count).to eq(n_outputs)
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.05).of(1.0)
    end
  end

  context 'when multiple regression problem with bias' do
    let(:y) { multi_target + Numo::DFloat[2, 1] }
    let(:fit_bias) { true }

    it 'learns the model for multiple regression problem with bias term', :aggregate_failures do
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
      expect(estimator.weight_vec.lt(0).count).to be_zero
      expect(estimator.bias_term).to be_a(Numo::DFloat)
      expect(estimator.bias_term).to be_contiguous
      expect(estimator.bias_term.ndim).to eq(1)
      expect(estimator.bias_term.shape[0]).to eq(n_outputs)
      expect(estimator.bias_term.ge(0).count).to eq(n_outputs)
      expect(score).to be_within(0.01).of(1.0)
    end
  end
end
