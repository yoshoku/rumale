# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::SVC do
  let(:x_bin) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y_bin) { Marshal.load(File.read(__dir__ + '/../test_labels.dat')) }
  let(:x_mlt) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:y_mlt) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:estimator) { described_class.new(random_seed: 1) }
  let(:estimator_prob) { described_class.new(probability: true, random_seed: 1) }
  let(:estimator_bias) { described_class.new(fit_bias: true, random_seed: 1) }
  let(:estimator_parallel) { described_class.new(fit_bias: true, probability: true, n_jobs: -1, random_seed: 1) }

  it 'classifies two clusters.' do
    n_samples, n_features = x_bin.shape
    estimator.fit(x_bin, y_bin)

    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(2)
    expect(estimator.classes.shape[0]).to eq(2)
    expect(estimator.classes.shape[1]).to be_nil

    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.size).to eq(n_features)
    expect(estimator.weight_vec.shape[0]).to eq(n_features)
    expect(estimator.weight_vec.shape[1]).to be_nil
    expect(estimator.bias_term).to be_zero

    func_vals = estimator.decision_function(x_bin)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to be_nil

    predicted = estimator.predict(x_bin)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil

    expect(predicted).to eq(y_bin)
    expect(estimator.score(x_bin, y_bin)).to eq(1.0)
  end

  it 'learns the model of two clusters dataset with bias term.' do
    _n_samples, n_features = x_bin.shape
    estimator_bias.fit(x_bin, y_bin)
    expect(estimator_bias.weight_vec.size).to eq(n_features)
    expect(estimator_bias.weight_vec.shape[0]).to eq(n_features)
    expect(estimator_bias.weight_vec.shape[1]).to be_nil
    expect(estimator_bias.bias_term).to_not be_zero
    expect(estimator_bias.score(x_bin, y_bin)).to eq(1.0)
  end

  it 'estimates class probabilities with two clusters dataset.' do
    n_samples, _n_features = x_bin.shape
    probs = estimator_prob.fit(x_bin, y_bin).predict_proba(x_bin)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(2)
    expect(probs.sum(1).eq(1).count).to eq(n_samples)
    predicted = Numo::Int32.cast(probs[true, 0] < probs[true, 1]) * 2 - 1
    expect(predicted).to eq(y_bin)
  end

  it 'classifies three clusters.' do
    n_classes = y_mlt.to_a.uniq.size
    n_samples, n_features = x_mlt.shape

    estimator_bias.fit(x_mlt, y_mlt)

    expect(estimator_bias.classes.class).to eq(Numo::Int32)
    expect(estimator_bias.classes.size).to eq(n_classes)
    expect(estimator_bias.classes.shape[0]).to eq(n_classes)
    expect(estimator_bias.classes.shape[1]).to be_nil

    expect(estimator_bias.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator_bias.weight_vec.size).to eq(n_classes * n_features)
    expect(estimator_bias.weight_vec.shape[0]).to eq(n_classes)
    expect(estimator_bias.weight_vec.shape[1]).to eq(n_features)

    expect(estimator_bias.bias_term.class).to eq(Numo::DFloat)
    expect(estimator_bias.bias_term.size).to eq(n_classes)
    expect(estimator_bias.bias_term.shape[0]).to eq(n_classes)
    expect(estimator_bias.bias_term.shape[1]).to be_nil

    predicted = estimator_bias.predict(x_mlt)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil

    expect(predicted).to eq(y_mlt)
    expect(estimator_bias.score(x_mlt, y_mlt)).to eq(1.0)
  end

  it 'estimates class probabilities with three clusters dataset.' do
    classes = y_mlt.to_a.uniq.sort
    n_classes = classes.size
    n_samples, _n_features = x_mlt.shape
    probs = estimator_prob.fit(x_mlt, y_mlt).predict_proba(x_mlt)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(n_classes)
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(predicted).to eq(y_mlt)
  end

  it 'estimates class probabilities with three clusters dataset in parallel.' do
    classes = y_mlt.to_a.uniq.sort
    n_classes = classes.size
    n_samples, n_features = x_mlt.shape
    estimator_parallel.fit(x_mlt, y_mlt)
    predicted = estimator_parallel.predict(x_mlt)
    probs = estimator_parallel.predict_proba(x_mlt)
    prob_predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(estimator_parallel.classes.class).to eq(Numo::Int32)
    expect(estimator_parallel.classes.size).to eq(n_classes)
    expect(estimator_parallel.classes.shape[0]).to eq(n_classes)
    expect(estimator_parallel.classes.shape[1]).to be_nil
    expect(estimator_parallel.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator_parallel.weight_vec.size).to eq(n_classes * n_features)
    expect(estimator_parallel.weight_vec.shape[0]).to eq(n_classes)
    expect(estimator_parallel.weight_vec.shape[1]).to eq(n_features)
    expect(estimator_parallel.bias_term.class).to eq(Numo::DFloat)
    expect(estimator_parallel.bias_term.size).to eq(n_classes)
    expect(estimator_parallel.bias_term.shape[0]).to eq(n_classes)
    expect(estimator_parallel.bias_term.shape[1]).to be_nil
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(predicted).to eq(y_mlt)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(n_classes)
    expect(prob_predicted).to eq(y_mlt)
    expect(estimator_parallel.score(x_mlt, y_mlt)).to eq(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x_bin, y_bin)
    copied = Marshal.load(Marshal.dump(estimator))
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
    expect(estimator.score(x_bin, y_bin)).to eq(copied.score(x_bin, y_bin))
  end
end
