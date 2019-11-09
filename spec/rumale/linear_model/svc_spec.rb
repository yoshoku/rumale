# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::SVC do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:fit_bias) { false }
  let(:probability) { false }
  let(:n_jobs) { nil }
  let(:estimator) { described_class.new(reg_param: 1, fit_bias: fit_bias, probability: probability, n_jobs: n_jobs, random_seed: 1).fit(x, y) }
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
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.weight_vec.class).to eq(Numo::DFloat)
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_features)
      expect(estimator.bias_term).to be_zero
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
      expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
      expect(estimator.params[:fit_bias]).to eq(copied.params[:fit_bias])
      expect(estimator.params[:bias_scale]).to eq(copied.params[:bias_scale])
      expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
      expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
      expect(estimator.params[:probability]).to eq(copied.params[:probability])
      expect(estimator.params[:optimizer].class).to eq(copied.params[:optimizer].class)
      expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
      expect(estimator.weight_vec).to eq(copied.weight_vec)
      expect(estimator.bias_term).to eq(copied.bias_term)
      expect(estimator.rng).to eq(copied.rng)
      expect(estimator.score(x, y)).to eq(copied.score(x, y))
    end

    context 'when fit_bias parameter is true' do
      let(:fit_bias) { true }

      it 'learns the model of two clusters dataset with bias term.', :aggregate_failures do
        expect(estimator.weight_vec.ndim).to eq(1)
        expect(estimator.weight_vec.shape[0]).to eq(n_features)
        expect(estimator.bias_term).not_to be_zero
        expect(score).to eq(1.0)
      end
    end

    context 'when probability parameter is true' do
      let(:probability) { true }

      it 'estimates class probabilities with two clusters dataset.', :aggregate_failures do
        expect(probs.class).to eq(Numo::DFloat)
        expect(probs.ndim).to eq(2)
        expect(probs.shape[0]).to eq(n_samples)
        expect(probs.shape[1]).to eq(n_classes)
        expect(probs.sum(1).eq(1).count).to eq(n_samples)
        expect(predicted_by_probs).to eq(y)
      end
    end
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }

    context 'when fit_bias parameter is true' do
      let(:fit_bias) { true }

      it 'classifies three clusters.', :aggregate_failures do
        expect(estimator.classes.class).to eq(Numo::Int32)
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(estimator.weight_vec.class).to eq(Numo::DFloat)
        expect(estimator.weight_vec.ndim).to eq(2)
        expect(estimator.weight_vec.shape[0]).to eq(n_classes)
        expect(estimator.weight_vec.shape[1]).to eq(n_features)
        expect(estimator.bias_term.class).to eq(Numo::DFloat)
        expect(estimator.bias_term.ndim).to eq(1)
        expect(estimator.bias_term.shape[0]).to eq(n_classes)
        expect(predicted.class).to eq(Numo::Int32)
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted).to eq(y)
        expect(score).to eq(1.0)
      end
    end

    context 'when probability parameter is true' do
      let(:probability) { true }

      it 'estimates class probabilities with three clusters dataset.', :aggregate_failures do
        expect(probs.class).to eq(Numo::DFloat)
        expect(probs.ndim).to eq(2)
        expect(probs.shape[0]).to eq(n_samples)
        expect(probs.shape[1]).to eq(n_classes)
        expect(predicted_by_probs).to eq(y)
      end
    end

    context 'when n_jobs parameter is not nil' do
      let(:probability) { true }
      let(:fit_bias) { true }
      let(:n_jobs) { -1 }

      it 'estimates class probabilities with three clusters dataset in parallel.', :aggregate_failures do
        expect(estimator.classes.class).to eq(Numo::Int32)
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(estimator.weight_vec.class).to eq(Numo::DFloat)
        expect(estimator.weight_vec.ndim).to eq(2)
        expect(estimator.weight_vec.shape[0]).to eq(n_classes)
        expect(estimator.weight_vec.shape[1]).to eq(n_features)
        expect(estimator.bias_term.class).to eq(Numo::DFloat)
        expect(estimator.bias_term.ndim).to eq(1)
        expect(estimator.bias_term.shape[0]).to eq(n_classes)
        expect(predicted.class).to eq(Numo::Int32)
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted).to eq(y)
        expect(probs.class).to eq(Numo::DFloat)
        expect(probs.ndim).to eq(2)
        expect(probs.shape[0]).to eq(n_samples)
        expect(probs.shape[1]).to eq(n_classes)
        expect(predicted_by_probs).to eq(y)
        expect(score).to eq(1.0)
      end
    end
  end
end
