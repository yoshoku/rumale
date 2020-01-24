# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::PolynomialModel::FactorizationMachineRegressor do
  let(:x) { two_clusters_dataset[0] }
  let(:single_target) { x.dot(Numo::DFloat[0.8, 0.2]) }
  let(:multi_target) { x.dot(Numo::DFloat[[0.8, 0.82], [0.2, 0.18]]) }
  let(:n_factors) { 2 }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_outputs) { multi_target.shape[1] }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(n_factors: n_factors, reg_param_linear: 0.1, reg_param_factor: 0.1,
                        n_jobs: n_jobs, random_seed: 1).fit(x, y)
  end
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when single target problem' do
    let(:y) { single_target }

    it 'learns the the model for single regression problem.', :aggregate_failures do
      expect(estimator.factor_mat.class).to eq(Numo::DFloat)
      expect(estimator.factor_mat.ndim).to eq(2)
      expect(estimator.factor_mat.shape[0]).to eq(n_factors)
      expect(estimator.factor_mat.shape[1]).to eq(n_features)
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.bias_term.class).to eq(Float)
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.factor_mat).to eq(copied.factor_mat)
      expect(estimator.weight_vec).to eq(copied.weight_vec)
      expect(estimator.bias_term).to eq(copied.bias_term)
      expect(estimator.rng).to eq(copied.rng)
      expect(estimator.params[:n_factors]).to eq(copied.params[:n_factors])
      expect(estimator.params[:reg_param_linear]).to eq(copied.params[:reg_param_linear])
      expect(estimator.params[:reg_param_factor]).to eq(copied.params[:reg_param_factor])
      expect(estimator.params[:learning_rate]).to eq(copied.params[:learning_rate])
      expect(estimator.params[:decay]).to eq(copied.params[:decay])
      expect(estimator.params[:momentum]).to eq(copied.params[:momentum])
      expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
      expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
      expect(estimator.params[:optimizer].class).to eq(copied.params[:optimizer].class)
      expect(estimator.params[:n_jobs]).to eq(copied.params[:n_jobs])
      expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
      expect(score).to eq(copied.score(x, y))
    end

    context 'when verbose is "true"' do
      let(:estimator) { described_class.new(verbose: true, max_iter: 5, random_seed: 1) }

      it 'outputs debug messages.', :aggregate_failures do
        expect { estimator.fit(x, y) }.to output(/FactorizationMachineRegressor/).to_stdout
      end
    end
  end

  context 'when multi-target problem' do
    let(:y) { multi_target }

    it 'learns the model for multiple-regression problem.', :aggregate_failures do
      expect(estimator.factor_mat.class).to eq(Numo::DFloat)
      expect(estimator.factor_mat.ndim).to eq(3)
      expect(estimator.factor_mat.shape[0]).to eq(n_outputs)
      expect(estimator.factor_mat.shape[1]).to eq(n_factors)
      expect(estimator.factor_mat.shape[2]).to eq(n_features)
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

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'learns the model for multiple-regression problem in parallel.', :aggregate_failures do
        expect(estimator.factor_mat.class).to eq(Numo::DFloat)
        expect(estimator.factor_mat.ndim).to eq(3)
        expect(estimator.factor_mat.shape[0]).to eq(n_outputs)
        expect(estimator.factor_mat.shape[1]).to eq(n_factors)
        expect(estimator.factor_mat.shape[2]).to eq(n_features)
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
