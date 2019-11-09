# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::PolynomialModel::FactorizationMachineClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:n_factors) { 2 }
  let(:loss) { 'hinge' }
  let(:reg_param_linear) { 0.1 }
  let(:reg_param_factor) { 0.1 }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(n_factors: n_factors, loss: loss, reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor,
                        n_jobs: n_jobs, random_seed: 1).fit(x, y)
  end
  let(:func_vals) { estimator.decision_function(x) }
  let(:predicted) { estimator.predict(x) }
  let(:probs) { estimator.predict_proba(x) }
  let(:score) { estimator.score(x, y) }
  let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when binary classification problem' do
    let(:dataset) { two_clusters_dataset }

    it 'classifies two clusters.', :aggregate_failures do
      expect(estimator.classes.class).to eq(Numo::Int32)
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(2)
      expect(estimator.factor_mat.class).to eq(Numo::DFloat)
      expect(estimator.factor_mat.ndim).to eq(2)
      expect(estimator.factor_mat.shape[0]).to eq(n_factors)
      expect(estimator.factor_mat.shape[1]).to eq(n_features)
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.bias_term.class).to eq(Float)
      expect(func_vals.class).to eq(Numo::DFloat)
      expect(func_vals.ndim).to eq(1)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(score).to eq(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.factor_mat).to eq(copied.factor_mat)
      expect(estimator.weight_vec).to eq(copied.weight_vec)
      expect(estimator.bias_term).to eq(copied.bias_term)
      expect(estimator.rng).to eq(copied.rng)
      expect(estimator.params[:n_factors]).to eq(copied.params[:n_factors])
      expect(estimator.params[:loss]).to eq(copied.params[:loss])
      expect(estimator.params[:reg_param_linear]).to eq(copied.params[:reg_param_linear])
      expect(estimator.params[:reg_param_factor]).to eq(copied.params[:reg_param_factor])
      expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
      expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
      expect(estimator.params[:optimizer].class).to eq(copied.params[:optimizer].class)
      expect(estimator.params[:n_jobs]).to eq(copied.params[:n_jobs])
      expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
      expect(score).to eq(copied.score(x, y))
    end

    context 'when loss parameter is "logit"' do
      let(:loss) { 'logit' }
      let(:reg_param_linear) { 0.001 }
      let(:reg_param_factor) { 0.01 }

      it 'estimates class probabilities with two clusters dataset.', :aggregate_failures do
        expect(probs.class).to eq(Numo::DFloat)
        expect(probs.ndim).to eq(2)
        expect(probs.shape[0]).to eq(n_samples)
        expect(probs.shape[1]).to eq(2)
        expect(probs.sum(1).eq(1).count).to eq(n_samples)
        expect(predicted_by_probs).to eq(y)
      end
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }

    it 'classifies three clusters.', :aggregate_failures do
      expect(estimator.classes.class).to eq(Numo::Int32)
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.factor_mat.class).to eq(Numo::DFloat)
      expect(estimator.factor_mat.ndim).to eq(3)
      expect(estimator.factor_mat.shape[0]).to eq(n_classes)
      expect(estimator.factor_mat.shape[1]).to eq(n_factors)
      expect(estimator.factor_mat.shape[2]).to eq(n_features)
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_classes)
      expect(estimator.weight_vec.shape[1]).to eq(n_features)
      expect(estimator.bias_term.class).to eq(Numo::DFloat)
      expect(estimator.bias_term.ndim).to eq(1)
      expect(estimator.bias_term.shape[0]).to eq(n_classes)
      expect(func_vals.class).to eq(Numo::DFloat)
      expect(func_vals.ndim).to eq(2)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(func_vals.shape[1]).to eq(n_classes)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(score).to eq(1.0)
    end

    context 'when loss parameter is "logit"' do
      let(:loss) { 'logit' }
      let(:reg_param_linear) { 0.001 }
      let(:reg_param_factor) { 0.01 }

      it 'estimates class probabilities with three clusters dataset.', :aggregate_failures do
        expect(probs.class).to eq(Numo::DFloat)
        expect(probs.ndim).to eq(2)
        expect(probs.shape[0]).to eq(n_samples)
        expect(probs.shape[1]).to eq(n_classes)
        expect(predicted_by_probs).to eq(y)
      end
    end

    context 'when n_jobs parameter is not nil' do
      let(:n_jobs) { -1 }

      it 'classifies three clusters in parallel.', :aggregate_failures do
        expect(estimator.classes.class).to eq(Numo::Int32)
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(estimator.factor_mat.class).to eq(Numo::DFloat)
        expect(estimator.factor_mat.ndim).to eq(3)
        expect(estimator.factor_mat.shape[0]).to eq(n_classes)
        expect(estimator.factor_mat.shape[1]).to eq(n_factors)
        expect(estimator.factor_mat.shape[2]).to eq(n_features)
        expect(estimator.weight_vec.class).to eq(Numo::DFloat)
        expect(estimator.weight_vec.ndim).to eq(2)
        expect(estimator.weight_vec.shape[0]).to eq(n_classes)
        expect(estimator.weight_vec.shape[1]).to eq(n_features)
        expect(estimator.bias_term.class).to eq(Numo::DFloat)
        expect(estimator.bias_term.ndim).to eq(1)
        expect(estimator.bias_term.shape[0]).to eq(n_classes)
        expect(func_vals.class).to eq(Numo::DFloat)
        expect(func_vals.ndim).to eq(2)
        expect(func_vals.shape[0]).to eq(n_samples)
        expect(func_vals.shape[1]).to eq(n_classes)
        expect(predicted.class).to eq(Numo::Int32)
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted).to eq(y)
        expect(score).to eq(1.0)
      end
    end
  end
end
