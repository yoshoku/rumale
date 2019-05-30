# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::CrossValidation do
  let(:samples) { Marshal.load(File.read(__dir__ + '/../test_samples_xor.dat')) }
  let(:labels) { Marshal.load(File.read(__dir__ + '/../test_labels_xor.dat')) }
  let(:values) { samples.dot(Numo::DFloat[1.0, 2.0]) }
  let(:kernel_mat) { Rumale::PairwiseMetric.rbf_kernel(samples, nil, 1.0) }
  let(:kernel_svc) { Rumale::KernelMachine::KernelSVC.new(reg_param: 1.0, max_iter: 1000, random_seed: 1) }
  let(:linear_svc) { Rumale::LinearModel::SVC.new(random_seed: 1) }
  let(:linear_svr) { Rumale::LinearModel::SVR.new(random_seed: 1) }
  let(:logit_reg) { Rumale::LinearModel::LogisticRegression.new(random_seed: 1) }
  let(:f_score) { Rumale::EvaluationMeasure::FScore.new }
  let(:log_loss) { Rumale::EvaluationMeasure::LogLoss.new }
  let(:n_splits) { 5 }
  let(:kfold) { Rumale::ModelSelection::KFold.new(n_splits: n_splits, shuffle: true, random_seed: 1) }
  let(:skfold) { Rumale::ModelSelection::StratifiedKFold.new(n_splits: n_splits, shuffle: true, random_seed: 1) }

  it 'performs k-fold cross validation with linear svc.' do
    cv = described_class.new(estimator: linear_svc, splitter: kfold)
    report = cv.perform(samples, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs k-fold cross validation with kernel svc.' do
    cv = described_class.new(estimator: kernel_svc, splitter: kfold)
    report = cv.perform(kernel_mat, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs stratified k-fold cross validation with kernel svc.' do
    cv = described_class.new(estimator: kernel_svc, splitter: skfold)
    report = cv.perform(kernel_mat, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs k-fold cross validation with linear svr.' do
    cv = described_class.new(estimator: linear_svr, splitter: kfold)
    report = cv.perform(samples, values)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'also calculates scores of training dataset.' do
    cv = described_class.new(estimator: kernel_svc, splitter: skfold, return_train_score: true)
    report = cv.perform(kernel_mat, labels)
    mean_test_score = report[:test_score].inject(:+) / n_splits
    mean_train_score = report[:train_score].inject(:+) / n_splits
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(0.1).of(0.9)
    expect(mean_train_score).to eq(1.0)
  end

  it 'performs k-fold cross validation with kernel svc to evaluate the results using F1-score.' do
    cv = described_class.new(estimator: kernel_svc, splitter: kfold, evaluator: f_score, return_train_score: true)
    report = cv.perform(kernel_mat, labels)
    mean_test_score = report[:test_score].inject(:+) / n_splits
    mean_train_score = report[:train_score].inject(:+) / n_splits
    expect(cv.evaluator.class).to eq(Rumale::EvaluationMeasure::FScore)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(0.1).of(0.9)
    expect(mean_train_score).to eq(1.0)
  end

  it 'performs k-fold cross validation with logistic regression to evaluate the results using Log-loss.' do
    cv = described_class.new(estimator: logit_reg, splitter: kfold, evaluator: log_loss, return_train_score: true)
    report = cv.perform(kernel_mat, labels)
    mean_test_score = report[:test_score].inject(:+) / n_splits
    mean_train_score = report[:train_score].inject(:+) / n_splits
    expect(cv.evaluator.class).to eq(Rumale::EvaluationMeasure::LogLoss)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(5.0e-5).of(0.8479)
    expect(mean_train_score).to be_within(5.0e-5).of(0.8541)
  end

  describe 'private method' do
    let(:ovr_kernel_svc) { Rumale::Multiclass::OneVsRestClassifier.new(estimator: kernel_svc) }
    let(:ovr_linear_svc) { Rumale::Multiclass::OneVsRestClassifier.new(estimator: linear_svc) }
    let(:kernel_svc_cv) { described_class.new(estimator: kernel_svc, splitter: kfold) }
    let(:linear_svc_cv) { described_class.new(estimator: linear_svc, splitter: kfold) }
    let(:ovr_kernel_svc_cv) { described_class.new(estimator: ovr_kernel_svc, splitter: kfold) }
    let(:ovr_linear_svc_cv) { described_class.new(estimator: ovr_linear_svc, splitter: kfold) }

    it 'detects type of classifier.' do
      expect(kernel_svc_cv.send(:kernel_machine?)).to be_truthy
      expect(linear_svc_cv.send(:kernel_machine?)).to be_falsey
      expect(ovr_kernel_svc_cv.send(:kernel_machine?)).to be_truthy
      expect(ovr_linear_svc_cv.send(:kernel_machine?)).to be_falsey
    end
  end
end
