# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::CrossValidation do
  let(:xor) { xor_dataset }
  let(:samples) { xor[0] }
  let(:labels) { xor[1] }
  let(:values) { samples.dot(Numo::DFloat[1.0, 2.0]) }
  let(:kernel_mat) { Rumale::PairwiseMetric.rbf_kernel(samples, nil, 1.0) }
  let(:kernel_svc) { Rumale::KernelMachine::KernelSVC.new(reg_param: 1.0, max_iter: 1000, random_seed: 1) }
  let(:linear_svc) { Rumale::LinearModel::SVC.new(random_seed: 1) }
  let(:linear_svr) { Rumale::LinearModel::SVR.new(random_seed: 1) }
  let(:logit_reg) { Rumale::LinearModel::LogisticRegression.new }
  let(:f_score) { Rumale::EvaluationMeasure::FScore.new }
  let(:log_loss) { Rumale::EvaluationMeasure::LogLoss.new }
  let(:n_splits) { 5 }
  let(:kfold) { Rumale::ModelSelection::KFold.new(n_splits: n_splits, shuffle: true, random_seed: 1) }
  let(:skfold) { Rumale::ModelSelection::StratifiedKFold.new(n_splits: n_splits, shuffle: true, random_seed: 1) }

  it 'performs k-fold cross validation with linear svc', :aggregate_failures do
    cv = described_class.new(estimator: linear_svc, splitter: kfold)
    report = cv.perform(samples, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs k-fold cross validation with kernel svc', :aggregate_failures do
    cv = described_class.new(estimator: kernel_svc, splitter: kfold)
    report = cv.perform(kernel_mat, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs stratified k-fold cross validation with kernel svc', :aggregate_failures do
    cv = described_class.new(estimator: kernel_svc, splitter: skfold)
    report = cv.perform(kernel_mat, labels)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'performs k-fold cross validation with linear svr', :aggregate_failures do
    cv = described_class.new(estimator: linear_svr, splitter: kfold)
    report = cv.perform(samples, values)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score]).to be_nil
    expect(report[:fit_time].size).to eq(n_splits)
  end

  it 'also calculates scores of training dataset', :aggregate_failures do
    cv = described_class.new(estimator: kernel_svc, splitter: skfold, return_train_score: true)
    report = cv.perform(kernel_mat, labels)
    mean_test_score = report[:test_score].sum / n_splits
    mean_train_score = report[:train_score].sum / n_splits
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(0.1).of(0.9)
    expect(mean_train_score).to eq(1.0)
  end

  it 'performs k-fold cross validation with kernel svc to evaluate the results using F1-score', :aggregate_failures do
    cv = described_class.new(estimator: kernel_svc, splitter: kfold, evaluator: f_score, return_train_score: true)
    report = cv.perform(kernel_mat, labels)
    mean_test_score = report[:test_score].sum / n_splits
    mean_train_score = report[:train_score].sum / n_splits
    expect(cv.evaluator).to be_a(Rumale::EvaluationMeasure::FScore)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(0.1).of(0.9)
    expect(mean_train_score).to eq(1.0)
  end

  it 'performs k-fold cross validation with logistic regression to evaluate the results using Log-loss',
     :aggregate_failures do
    cv = described_class.new(estimator: logit_reg, splitter: kfold, evaluator: log_loss, return_train_score: true)
    report = cv.perform(samples, labels)
    mean_test_score = report[:test_score].sum / n_splits
    mean_train_score = report[:train_score].sum / n_splits
    expect(cv.evaluator).to be_a(Rumale::EvaluationMeasure::LogLoss)
    expect(report[:test_score].size).to eq(n_splits)
    expect(report[:train_score].size).to eq(n_splits)
    expect(report[:fit_time].size).to eq(n_splits)
    expect(mean_test_score).to be_within(0.01).of(0.7)
    expect(mean_train_score).to be_within(0.01).of(0.7)
  end

  describe 'private method' do
    let(:kernel_svc_cv) { described_class.new(estimator: kernel_svc, splitter: kfold) }
    let(:linear_svc_cv) { described_class.new(estimator: linear_svc, splitter: kfold) }

    it 'detects type of classifier', :aggregate_failures do
      expect(kernel_svc_cv.send(:kernel_machine?)).to be_truthy
      expect(linear_svc_cv.send(:kernel_machine?)).to be_falsey
    end
  end
end
