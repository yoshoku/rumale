# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::KernelMachine::KernelSVC do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_classes) { classes.size }
  let(:kernel_mat) { Rumale::PairwiseMetric.rbf_kernel(x, nil, 1.0) }
  let(:probability) { false }
  let(:n_jobs) { nil }
  let(:estimator) do
    described_class.new(reg_param: 1, max_iter: 1000, probability: probability,
                        n_jobs: n_jobs, random_seed: 1).fit(kernel_mat, y)
  end
  let(:func_vals) { estimator.decision_function(kernel_mat) }
  let(:predicted) { estimator.predict(kernel_mat) }
  let(:probs) { estimator.predict_proba(kernel_mat) }
  let(:score) { estimator.score(kernel_mat, y) }
  let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  context 'when binary classification problem' do
    let(:dataset) { xor_dataset }

    it 'classifies xor data.', :aggregate_failures do
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_samples)
      expect(func_vals).to be_a(Numo::DFloat)
      expect(func_vals).to be_contiguous
      expect(func_vals.ndim).to eq(1)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(score).to eq(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
      expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
      expect(estimator.params[:probability]).to eq(copied.params[:probability])
      expect(estimator.params[:n_jobs]).to eq(copied.params[:n_jobs])
      expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
      expect(estimator.weight_vec).to eq(copied.weight_vec)
      expect(estimator.rng).to eq(copied.rng)
      expect(score).to eq(copied.score(kernel_mat, y))
    end

    context 'when probability parameter is true' do
      let(:probability) { true }

      it 'estimates class probabilities with xor data.', :aggregate_failures do
        expect(probs).to be_a(Numo::DFloat)
        expect(probs).to be_contiguous
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
      expect(estimator.classes).to be_a(Numo::Int32)
      expect(estimator.classes).to be_contiguous
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_classes)
      expect(estimator.weight_vec.shape[1]).to eq(n_samples)
      expect(func_vals).to be_a(Numo::DFloat)
      expect(func_vals).to be_contiguous
      expect(func_vals.ndim).to eq(2)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(func_vals.shape[1]).to eq(n_classes)
      expect(predicted).to be_a(Numo::Int32)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(score).to eq(1.0)
    end

    context 'when probability parameter is true' do
      let(:probability) { true }

      it 'estimates class probabilities with three clusters dataset.', :aggregate_failures do
        expect(probs).to be_a(Numo::DFloat)
        expect(probs).to be_contiguous
        expect(probs.ndim).to eq(2)
        expect(probs.shape[0]).to eq(n_samples)
        expect(probs.shape[1]).to eq(n_classes)
        expect(predicted_by_probs).to eq(y)
      end
    end

    context 'when n_jobs is not nil' do
      let(:probability) { true }
      let(:n_jobs) { -1 }

      it 'estimates class probabilities with three clusters dataset in parallel.', :aggregate_failures do
        # FIXME: Remove Numo::Linalg temporarily for avoiding Parallel::DeadWorker error.
        backup = Numo::Linalg
        Numo.class_eval { remove_const(:Linalg) }

        expect(estimator.classes).to be_a(Numo::Int32)
        expect(estimator.classes).to be_contiguous
        expect(estimator.classes.ndim).to eq(1)
        expect(estimator.classes.shape[0]).to eq(n_classes)
        expect(estimator.weight_vec).to be_a(Numo::DFloat)
        expect(estimator.weight_vec).to be_contiguous
        expect(estimator.weight_vec.ndim).to eq(2)
        expect(estimator.weight_vec.shape[0]).to eq(n_classes)
        expect(estimator.weight_vec.shape[1]).to eq(n_samples)
        expect(func_vals).to be_a(Numo::DFloat)
        expect(func_vals).to be_contiguous
        expect(func_vals.ndim).to eq(2)
        expect(func_vals.shape[0]).to eq(n_samples)
        expect(func_vals.shape[1]).to eq(n_classes)
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

        Numo::Linalg = backup
      end
    end
  end
end
