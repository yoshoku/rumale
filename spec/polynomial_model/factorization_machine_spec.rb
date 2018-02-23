require 'spec_helper'

RSpec.describe SVMKit::PolynomialModel::FactorizationMachineClassifier do
  let(:samples) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:labels) { Marshal.load(File.read(__dir__ + '/../test_labels.dat')) }
  let(:estimator) do
    described_class.new(n_factors: 2, reg_param_bias: 0.001, reg_param_weight: 0.001, reg_param_factor: 0.001,
                        init_std: 0.01, max_iter: 1000, batch_size: 10, random_seed: 1)
  end
  let(:estimator_logit) do
    described_class.new(n_factors: 2, loss: 'logistic',
                        reg_param_bias: 0.001, reg_param_weight: 0.001, reg_param_factor: 0.001,
                        init_std: 0.01, max_iter: 1000, batch_size: 10, random_seed: 1)
  end

  it 'classifies two clusters.' do
    n_samples, n_features = samples.shape
    estimator.fit(samples, labels)
    expect(estimator.weight_vec.class).to eq(Numo::DFloat)
    expect(estimator.weight_vec.shape[0]).to eq(n_features)
    expect(estimator.weight_vec.shape[1]).to be_nil

    func_vals = estimator.decision_function(samples)
    expect(func_vals.class).to eq(Numo::DFloat)
    expect(func_vals.shape[0]).to eq(n_samples)
    expect(func_vals.shape[1]).to be_nil

    predicted = estimator.predict(samples)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted.shape[1]).to be_nil
    expect(predicted).to eq(labels)

    score = estimator.score(samples, labels)
    expect(score).to eq(1.0)
  end

  it 'estimates class probabilities with two clusters dataset.' do
    n_samples, _n_features = samples.shape
    estimator_logit.fit(samples, labels)
    probs = estimator_logit.predict_proba(samples)
    expect(probs.class).to eq(Numo::DFloat)
    expect(probs.shape[0]).to eq(n_samples)
    expect(probs.shape[1]).to eq(2)
    expect(probs.sum(1).eq(1).count).to eq(n_samples)
    predicted = Numo::Int32.cast(probs[true, 0] < probs[true, 1]) * 2 - 1
    expect(predicted).to eq(labels)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(samples, labels)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.factor_mat).to eq(copied.factor_mat)
    expect(estimator.weight_vec).to eq(copied.weight_vec)
    expect(estimator.bias_term).to eq(copied.bias_term)
    expect(estimator.rng).to eq(copied.rng)
    expect(estimator.params[:n_factors]).to eq(copied.params[:n_factors])
    expect(estimator.params[:loss]).to eq(copied.params[:loss])
    expect(estimator.params[:reg_param_bias]).to eq(copied.params[:reg_param_bias])
    expect(estimator.params[:reg_param_weight]).to eq(copied.params[:reg_param_weight])
    expect(estimator.params[:reg_param_factor]).to eq(copied.params[:reg_param_factor])
    expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
    expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
  end
end
