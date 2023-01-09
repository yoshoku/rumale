# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::LinearRegression do
  let(:x) { two_clusters_dataset[0] }
  let(:single_target) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:multi_target) { x.dot(Numo::DFloat[[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { multi_target.shape[1] }
  let(:fit_bias) { false }
  let(:estimator) { described_class.new(fit_bias: fit_bias, solver: solver).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  shared_examples 'single regression' do
    let(:y) { single_target }

    it 'learns the model for single regression problem', :aggregate_failures do
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

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.class).to eq(estimator.class)
      expect(copied.params).to eq(estimator.params)
      expect(copied.weight_vec).to eq(estimator.weight_vec)
      expect(copied.bias_term).to eq(estimator.bias_term)
      expect(copied.score(x, y)).to eq(score)
    end
  end

  shared_examples 'single regression with bias' do
    let(:y) { single_target }
    let(:fit_bias) { true }

    it 'learns the model for single regression problem with bias term', :aggregate_failures do
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.bias_term).not_to be_zero
      expect(score).to be_within(0.01).of(1.0)
    end
  end

  shared_examples 'multiple regression' do
    let(:y) { multi_target }

    it 'learns the model for multiple-regression problems', :aggregate_failures do
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_outputs)
      expect(estimator.weight_vec.shape[1]).to eq(n_features)
      expect(estimator.bias_term).to be_a(Numo::DFloat)
      expect(estimator.bias_term).to be_contiguous
      expect(estimator.bias_term.ndim).to eq(1)
      expect(estimator.bias_term.shape[0]).to eq(n_outputs)
      expect(estimator.bias_term.to_a).to all(be_zero)
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.01).of(1.0)
    end
  end

  shared_examples 'multiple regression with bias' do
    let(:y) { multi_target }
    let(:fit_bias) { true }

    it 'learns the model for single regression problem with bias term', :aggregate_failures do
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_outputs)
      expect(estimator.weight_vec.shape[1]).to eq(n_features)
      expect(estimator.bias_term).to be_a(Numo::DFloat)
      expect(estimator.bias_term).to be_contiguous
      expect(estimator.bias_term.ndim).to eq(1)
      expect(estimator.bias_term.shape[0]).to eq(n_outputs)
      expect(estimator.bias_term.sum).not_to be_zero
      expect(Math.sqrt((estimator.bias_term**2).sum)).not_to be_zero
      expect(score).to be_within(0.01).of(1.0)
    end
  end

  context 'when solver is singular value decomposition' do
    let(:solver) { 'svd' }

    it_behaves_like 'single regression'
    it_behaves_like 'single regression with bias'
    it_behaves_like 'multiple regression with bias'
    it_behaves_like 'multiple regression'
  end

  context 'when solver is l-bfgs' do
    let(:solver) { 'lbfgs' }

    it_behaves_like 'single regression'
    it_behaves_like 'single regression with bias'
    it_behaves_like 'multiple regression with bias'
    it_behaves_like 'multiple regression'
  end

  context 'when solver is automatic' do
    let(:estimator) { described_class.new(solver: 'auto') }

    context 'with Numo::Linalg is loaded' do
      it 'chooses "svd" solver' do
        expect(estimator.params[:solver]).to eq('svd')
      end
    end

    context 'with Numo::Linalg is not loaded' do
      before { hide_const('Numo::Linalg') }

      it 'chooses "lbfgs" solver' do
        expect(estimator.params[:solver]).to eq('lbfgs')
      end
    end
  end
end
