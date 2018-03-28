# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::LinearModel::SVC do
  let(:x_bin) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y_bin) { Marshal.load(File.read(__dir__ + '/../test_labels.dat')) }
  let(:x_mlt) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:y_mlt) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:estimator) { described_class.new(random_seed: 1) }
  let(:estimator_bias) { described_class.new(fit_bias: true, random_seed: 1) }

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

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x_bin, y_bin)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
    expect(estimator.params[:fit_bias]).to eq(copied.params[:fit_bias])
    expect(estimator.params[:bias_scale]).to eq(copied.params[:bias_scale])
    expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
    expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(estimator.weight_vec).to eq(copied.weight_vec)
    expect(estimator.bias_term).to eq(copied.bias_term)
    expect(estimator.rng).to eq(copied.rng)
    expect(copied.score(x_bin, y_bin)).to eq(1.0)
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
end
