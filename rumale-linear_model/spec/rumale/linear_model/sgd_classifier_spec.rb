# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::SGDClassifier do
  let(:dataset) { two_clusters_dataset }
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { classes.size }
  let(:fit_bias) { false }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(loss: loss, reg_param: 1, fit_bias: fit_bias, n_jobs: n_jobs, random_seed: 1).fit(x, y)
  end
  let(:func_vals) { estimator.decision_function(x) }
  let(:predicted) { estimator.predict(x) }
  let(:probs) { estimator.predict_proba(x) }
  let(:score) { estimator.score(x, y) }
  let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  shared_examples 'classification problems' do
    context 'when binary classification problem' do
      it 'classifies two clusters', :aggregate_failures do
        expect(estimator.classes).to be_a(Numo::Int32)
        expect(estimator.classes).to be_contiguous
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(estimator.weight_vec).to be_a(Numo::DFloat)
        expect(estimator.weight_vec).to be_contiguous
        expect(estimator.weight_vec.ndim).to eq(1)
        expect(estimator.weight_vec.shape[0]).to eq(n_features)
        expect(estimator.bias_term).to be_zero
        expect(func_vals).to be_a(Numo::DFloat)
        expect(func_vals).to be_contiguous
        expect(func_vals.ndim).to eq(1)
        expect(func_vals.shape[0]).to eq(n_samples)
        expect(probs).to be_a(Numo::DFloat)
        expect(probs).to be_contiguous
        expect(probs.ndim).to eq(2)
        expect(probs.shape[0]).to eq(n_samples)
        expect(probs.shape[1]).to eq(n_classes)
        expect(probs.sum(axis: 1).eq(1).count).to eq(n_samples)
        expect(predicted_by_probs).to eq(y)
        expect(predicted).to be_a(Numo::Int32)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(1)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted).to eq(y)
        expect(score).to eq(1.0)
      end

      context 'when fit_bias parameter is true' do
        let(:fit_bias) { true }

        it 'learns the model of two clusters dataset with bias term', :aggregate_failures do
          expect(estimator.weight_vec.ndim).to eq(1)
          expect(estimator.weight_vec.shape[0]).to eq(n_features)
          expect(estimator.bias_term).not_to be_zero
          expect(score).to eq(1.0)
        end
      end
    end

    context 'when multiclass classification problem' do
      let(:dataset) { three_clusters_dataset }

      context 'when fit_bias parameter is true' do
        let(:fit_bias) { true }

        it 'classifies three clusters', :aggregate_failures do
          expect(estimator.classes).to be_a(Numo::Int32)
          expect(estimator.classes).to be_contiguous
          expect(estimator.classes.ndim).to eq(1)
          expect(estimator.classes.shape[0]).to eq(n_classes)
          expect(estimator.weight_vec).to be_a(Numo::DFloat)
          expect(estimator.weight_vec).to be_contiguous
          expect(estimator.weight_vec.ndim).to eq(2)
          expect(estimator.weight_vec.shape[0]).to eq(n_classes)
          expect(estimator.weight_vec.shape[1]).to eq(n_features)
          expect(estimator.bias_term).to be_a(Numo::DFloat)
          expect(estimator.bias_term).to be_contiguous
          expect(estimator.bias_term.ndim).to eq(1)
          expect(estimator.bias_term.shape[0]).to eq(n_classes)
          expect(probs).to be_a(Numo::DFloat)
          expect(probs).to be_contiguous
          expect(probs.ndim).to eq(2)
          expect(probs.shape[0]).to eq(n_samples)
          expect(probs.shape[1]).to eq(n_classes)
          expect(predicted_by_probs).to eq(y)
          expect(predicted).to be_a(Numo::Int32)
          expect(predicted).to be_contiguous
          expect(predicted.ndim).to eq(1)
          expect(predicted.shape[0]).to eq(n_samples)
          expect(predicted).to eq(y)
          expect(score).to eq(1.0)
        end
      end

      context 'when n_jobs parameter is not nil' do
        let(:fit_bias) { true }
        let(:n_jobs) { -1 }

        it 'estimates class probabilities with three clusters dataset in parallel', :aggregate_failures do
          expect(estimator.classes).to be_a(Numo::Int32)
          expect(estimator.classes).to be_contiguous
          expect(estimator.classes.ndim).to eq(1)
          expect(estimator.classes.shape[0]).to eq(n_classes)
          expect(estimator.weight_vec).to be_a(Numo::DFloat)
          expect(estimator.weight_vec).to be_contiguous
          expect(estimator.weight_vec.ndim).to eq(2)
          expect(estimator.weight_vec.shape[0]).to eq(n_classes)
          expect(estimator.weight_vec.shape[1]).to eq(n_features)
          expect(estimator.bias_term).to be_a(Numo::DFloat)
          expect(estimator.bias_term).to be_contiguous
          expect(estimator.bias_term.ndim).to eq(1)
          expect(estimator.bias_term.shape[0]).to eq(n_classes)
          expect(predicted).to be_a(Numo::Int32)
          expect(predicted).to be_contiguous
          expect(predicted.ndim).to eq(1)
          expect(predicted.shape[0]).to eq(n_samples)
          expect(predicted).to eq(y)
          expect(probs).to be_a(Numo::DFloat)
          expect(probs).to be_contiguous
          expect(probs.ndim).to eq(2)
          expect(probs.shape[0]).to eq(n_samples)
          expect(probs.shape[1]).to eq(n_classes)
          expect(predicted_by_probs).to eq(y)
          expect(score).to eq(1.0)
        end
      end
    end
  end

  context 'when loss is "hinge"' do
    let(:loss) { 'hinge' }

    it_behaves_like 'classification problems'

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.class).to eq(estimator.class)
      expect(copied.params).to eq(estimator.params)
      expect(copied.weight_vec).to eq(estimator.weight_vec)
      expect(copied.bias_term).to eq(estimator.bias_term)
      expect(copied.rng).to eq(estimator.rng)
      expect(copied.score(x, y)).to eq(score)
      expect(copied.instance_variable_get(:@penalty_type)).to eq('l2')
      expect(copied.instance_variable_get(:@loss_func)).to be_a(Rumale::LinearModel::Loss::HingeLoss)
    end
  end

  context 'when loss is "log_loss"' do
    let(:loss) { 'log_loss' }

    it_behaves_like 'classification problems'

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied.class).to eq(estimator.class)
      expect(copied.params).to eq(estimator.params)
      expect(copied.weight_vec).to eq(estimator.weight_vec)
      expect(copied.bias_term).to eq(estimator.bias_term)
      expect(copied.rng).to eq(estimator.rng)
      expect(copied.score(x, y)).to eq(score)
      expect(copied.instance_variable_get(:@penalty_type)).to eq('l2')
      expect(copied.instance_variable_get(:@loss_func)).to be_a(Rumale::LinearModel::Loss::LogLoss)
    end
  end

  context 'when loss is "log"' do
    let(:loss) { 'log' }

    it 'raises ArgumentError' do
      expect { estimator }.to raise_error(ArgumentError, /given loss 'log' is not supported./)
    end
  end
end
