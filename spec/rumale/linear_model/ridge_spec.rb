# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Ridge do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:single_target) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:multi_target) { x.dot(Numo::DFloat[[1.0, 2.0], [2.0, 1.0]]) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { multi_target.shape[1] }
  let(:fit_bias) { false }
  let(:solver) { 'sgd' }
  let(:n_jobs) { nil }
  let(:estimator) { described_class.new(reg_param: 0.1, fit_bias: fit_bias, solver: solver, n_jobs: n_jobs, random_seed: 1).fit(x, y) }
  let(:predicted) { estimator.predict(x) }

  shared_examples 'single regression' do
    let(:y) { single_target }

    it 'learns the model for single regression problem.', aggregate_failures: true do
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.size).to eq(n_features)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.shape[1]).to be_nil
      expect(estimator.bias_term).to be_zero
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to be_nil
      expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
    end

    it 'dumps and restores itself using Marshal module.', aggregate_failures: true do
      copied = Marshal.load(Marshal.dump(estimator))
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
      expect(estimator.params[:fit_bias]).to eq(copied.params[:fit_bias])
      expect(estimator.params[:bias_scale]).to eq(copied.params[:bias_scale])
      expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
      expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
      expect(estimator.params[:optimizer].class).to eq(copied.params[:optimizer].class)
      expect(estimator.params[:solver]).to eq(copied.params[:solver])
      expect(estimator.params[:n_jobs]).to eq(copied.params[:n_jobs])
      expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
      expect(estimator.weight_vec).to eq(copied.weight_vec)
      expect(estimator.bias_term).to eq(copied.bias_term)
      expect(estimator.rng).to eq(copied.rng)
      expect(estimator.score(x, y)).to eq(copied.score(x, y))
    end
  end

  shared_examples 'single regression with bias' do
    let(:y) { single_target }
    let(:fit_bias) { true }

    it 'learns the model for single regression problem with bias term.', aggregate_failures: true do
      expect(estimator.weight_vec.size).to eq(n_features)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.shape[1]).to be_nil
      expect(estimator.bias_term).not_to be_zero
      expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
    end
  end

  shared_examples 'multiple regression' do
    let(:y) { multi_target }

    it 'learns the model for multiple-regression problems.', aggregate_failures: true do
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.size).to eq(n_features * n_outputs)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
    end
  end

  shared_examples 'multiple regression with bias' do
    let(:y) { multi_target }
    let(:fit_bias) { true }

    it 'learns the model for single regression problem with bias term.', aggregate_failures: true do
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.size).to eq(n_features * n_outputs)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
      expect(estimator.bias_term.class).to eq(Numo::DFloat)
      expect(estimator.bias_term.size).to eq(n_outputs)
      expect(estimator.bias_term.shape[0]).to eq(n_outputs)
      expect(estimator.bias_term.shape[1]).to eq(nil)
      expect(Math.sqrt((estimator.bias_term**2).sum)).not_to be_zero
      expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
    end
  end

  shared_examples 'multiple regression with parallel' do
    let(:y) { multi_target }
    let(:n_jobs) { -1 }

    it 'learns the model for multiple-regression problems.', aggregate_failures: true do
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.size).to eq(n_features * n_outputs)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(estimator.score(x, y)).to be_within(0.01).of(1.0)
    end
  end

  context 'when solver is stochastic gradient descent' do
    let(:solver) { 'sgd' }

    it_behaves_like 'single regression'
    it_behaves_like 'single regression with bias'
    it_behaves_like 'multiple regression'
    it_behaves_like 'multiple regression with bias'
    it_behaves_like 'multiple regression with parallel'
  end

  context 'when solver is singular value decomposition' do
    let(:solver) { 'svd' }

    it_behaves_like 'single regression'
    it_behaves_like 'single regression with bias'
    it_behaves_like 'multiple regression with bias'
    it_behaves_like 'multiple regression'
  end
end
