# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::PolynomialModel::FactorizationMachineClassifier do
  let(:x_bin) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y_bin) { Marshal.load(File.read(__dir__ + '/../test_labels.dat')) }
  let(:x_mlt) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:y_mlt) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:n_factors) { 2 }
  let(:estimator) { described_class.new(n_factors: n_factors, reg_param_linear: 0.1, reg_param_factor: 0.1, random_seed: 1) }
  let(:estimator_logit) { described_class.new(n_factors: n_factors, loss: 'logistic', reg_param_linear: 0.001, reg_param_factor: 0.01, random_seed: 1) }

  it 'classifies two clusters.' do
    n_samples, n_features = x_bin.shape

    estimator.fit(x_bin, y_bin)

    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(2)
    expect(estimator.classes.shape[0]).to eq(2)
    expect(estimator.classes.shape[1]).to be_nil

    expect(estimator.factor_mat.class).to eq(Numo::DFloat)
    expect(estimator.factor_mat.size).to eq(n_factors * n_features)
    expect(estimator.factor_mat.shape[0]).to eq(n_factors)
    expect(estimator.factor_mat.shape[1]).to eq(n_features)

    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.size).to eq(n_features)
    expect(estimator.weight_vec.shape[0]).to eq(n_features)
    expect(estimator.weight_vec.shape[1]).to be_nil

    expect(estimator.bias_term.class).to eq(Float)

    func_vals = estimator.decision_function(x_bin)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to be_nil

    predicted = estimator.predict(x_bin)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil

    expect(estimator.score(x_bin, y_bin)).to eq(1.0)
    expect(predicted).to eq(y_bin)
  end

  it 'classifies three clusters.' do
    n_classes = y_mlt.to_a.uniq.size
    n_samples, n_features = x_mlt.shape

    estimator.fit(x_mlt, y_mlt)

    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.size).to eq(n_classes)
    expect(estimator.classes.shape[0]).to eq(n_classes)
    expect(estimator.classes.shape[1]).to be_nil

    expect(estimator.factor_mat.class).to eq(Numo::DFloat)
    expect(estimator.factor_mat.size).to eq(n_classes * n_factors * n_features)
    expect(estimator.factor_mat.shape[0]).to eq(n_classes)
    expect(estimator.factor_mat.shape[1]).to eq(n_factors)
    expect(estimator.factor_mat.shape[2]).to eq(n_features)

    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.size).to eq(n_classes * n_features)
    expect(estimator.weight_vec.shape[0]).to eq(n_classes)
    expect(estimator.weight_vec.shape[1]).to eq(n_features)

    expect(estimator.bias_term.class).to eq(Numo::DFloat)
    expect(estimator.bias_term.size).to eq(n_classes)
    expect(estimator.bias_term.shape[0]).to eq(n_classes)
    expect(estimator.bias_term.shape[1]).to be_nil

    func_vals = estimator.decision_function(x_mlt)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to eq(n_classes)

    predicted = estimator.predict(x_mlt)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil

    expect(predicted).to eq(y_mlt)
    expect(estimator.score(x_mlt, y_mlt)).to eq(1.0)
  end

  it 'estimates class probabilities with two clusters dataset.' do
    n_samples, _n_features = x_bin.shape
    estimator_logit.fit(x_bin, y_bin)
    probs = estimator_logit.predict_proba(x_bin)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(2)
    expect(probs.sum(1).eq(1).count).to eq(n_samples)
    predicted = Numo::Int32.cast(probs[true, 0] < probs[true, 1]) * 2 - 1
    expect(predicted).to eq(y_bin)
  end

  it 'estimates class probabilities with three clusters dataset.' do
    classes = y_mlt.to_a.uniq.sort
    n_classes = classes.size
    n_samples, _n_features = x_mlt.shape
    estimator_logit.fit(x_mlt, y_mlt)
    probs = estimator_logit.predict_proba(x_mlt)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(n_classes)
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(predicted).to eq(y_mlt)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(x_bin, y_bin)
    copied = Marshal.load(Marshal.dump(estimator))
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
    expect(estimator.params[:optimizer]).to eq(copied.params[:optimizer])
    expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(estimator.score(x_bin, y_bin)).to eq(copied.score(x_bin, y_bin))
  end
end
