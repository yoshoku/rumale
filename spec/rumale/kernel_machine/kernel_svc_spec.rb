# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::KernelMachine::KernelSVC do
  let(:xor) { xor_dataset }
  let(:x_xor) { xor[0] }
  let(:y_xor) { xor[1] }
  let(:three_clusters) { three_clusters_dataset }
  let(:x_mlt) { three_clusters[0] }
  let(:y_mlt) { three_clusters[1] }
  let(:kernel_mat_xor) { Rumale::PairwiseMetric.rbf_kernel(x_xor, nil, 1.0) }
  let(:kernel_mat_mlt) { Rumale::PairwiseMetric.rbf_kernel(x_mlt, nil, 1.0) }
  let(:estimator) { described_class.new(reg_param: 1.0, max_iter: 1000, random_seed: 1) }
  let(:estimator_prob) { described_class.new(reg_param: 1.0, max_iter: 1000, probability: true, random_seed: 1) }
  let(:estimator_parallel) { described_class.new(reg_param: 1.0, max_iter: 1000, probability: true, n_jobs: -1, random_seed: 1) }

  it 'classifies xor data.' do
    n_samples, = x_xor.shape[0]
    estimator.fit(kernel_mat_xor, y_xor)
    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.shape[0]).to eq(n_samples)
    expect(estimator.weight_vec.shape[1]).to be_nil

    func_vals = estimator.decision_function(kernel_mat_xor)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to be_nil

    predicted = estimator.predict(kernel_mat_xor)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil

    expect(predicted).to eq(y_xor)
    expect(estimator.score(kernel_mat_xor, y_xor)).to eq(1.0)
  end

  it 'classifies three clusters.' do
    n_classes = y_mlt.to_a.uniq.size
    n_samples, = x_mlt.shape

    estimator.fit(kernel_mat_mlt, y_mlt)

    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(n_classes)
    expect(estimator.classes.shape[0]).to eq(n_classes)
    expect(estimator.classes.shape[1]).to be_nil

    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.size).to eq(n_classes * n_samples)
    expect(estimator.weight_vec.shape[0]).to eq(n_classes)
    expect(estimator.weight_vec.shape[1]).to eq(n_samples)

    func_vals = estimator.decision_function(kernel_mat_mlt)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to eq(n_classes)

    predicted = estimator.predict(kernel_mat_mlt)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil

    expect(predicted).to eq(y_mlt)
    expect(estimator.score(kernel_mat_mlt, y_mlt)).to eq(1.0)
  end

  it 'estimates class probabilities with xor data.' do
    n_samples, = x_xor.shape[0]
    estimator_prob.fit(kernel_mat_xor, y_xor)
    probs = estimator_prob.predict_proba(kernel_mat_xor)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(2)
    expect(probs.sum(1).eq(1).count).to eq(n_samples)
    predicted = Numo::Int32.cast(probs[true, 0] < probs[true, 1]) * 2 - 1
    expect(predicted).to eq(y_xor)
  end

  it 'estimates class probabilities with three clusters dataset.' do
    classes = y_mlt.to_a.uniq.sort
    n_classes = classes.size
    n_samples = x_mlt.shape[0]
    estimator_prob.fit(kernel_mat_mlt, y_mlt)
    probs = estimator_prob.predict_proba(kernel_mat_mlt)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(n_classes)
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(predicted).to eq(y_mlt)
  end

  it 'estimates class probabilities with three clusters dataset in parallel.' do
    # FIXME: Remove Numo::Linalg temporarily for avoiding Parallel::DeadWorker error.
    backup = Numo::Linalg
    Numo.class_eval { remove_const(:Linalg) }

    classes = y_mlt.to_a.uniq.sort
    n_classes = classes.size
    n_samples = x_mlt.shape[0]
    estimator_parallel.fit(kernel_mat_mlt, y_mlt)
    func_vals = estimator_parallel.decision_function(kernel_mat_mlt)
    predicted = estimator_parallel.predict(kernel_mat_mlt)
    probs = estimator_parallel.predict_proba(kernel_mat_mlt)
    prob_predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(estimator_parallel.classes.class).to eq(Numo::Int32)
    expect(estimator_parallel.classes.size).to eq(n_classes)
    expect(estimator_parallel.classes.shape[0]).to eq(n_classes)
    expect(estimator_parallel.classes.shape[1]).to be_nil
    expect(estimator_parallel.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator_parallel.weight_vec.size).to eq(n_classes * n_samples)
    expect(estimator_parallel.weight_vec.shape[0]).to eq(n_classes)
    expect(estimator_parallel.weight_vec.shape[1]).to eq(n_samples)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to eq(n_classes)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(predicted).to eq(y_mlt)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(n_classes)
    expect(prob_predicted).to eq(y_mlt)
    expect(estimator_parallel.score(kernel_mat_mlt, y_mlt)).to eq(1.0)

    Numo::Linalg = backup
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(kernel_mat_xor, y_xor)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
    expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(estimator.params[:probability]).to eq(copied.params[:probability])
    expect(estimator.params[:n_jobs]).to eq(copied.params[:n_jobs])
    expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(estimator.weight_vec).to eq(copied.weight_vec)
    expect(estimator.rng).to eq(copied.rng)
    expect(copied.score(kernel_mat_xor, y_xor)).to eq(1.0)
  end
end
