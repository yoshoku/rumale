# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::ModelSelection::GridSearchCV do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:y) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:x_xor) { Marshal.load(File.read(__dir__ + '/../test_samples_xor.dat')) }
  let(:y_xor) { Marshal.load(File.read(__dir__ + '/../test_labels_xor.dat')) }
  let(:x_reg) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y_reg) { x_reg[true, 0] + x_reg[true, 1]**2 }
  let(:kfold) { SVMKit::ModelSelection::KFold.new(n_splits: 5, shuffle: true, random_seed: 1) }
  let(:skfold) { SVMKit::ModelSelection::StratifiedKFold.new(n_splits: 5, shuffle: true, random_seed: 1) }
  let(:rbf) { SVMKit::KernelApproximation::RBF.new(gamma: 0.1, random_seed: 1) }
  let(:svc) { SVMKit::LinearModel::SVC.new(random_seed: 1) }
  let(:scl) { SVMKit::Preprocessing::MinMaxScaler.new }
  let(:nbs) { SVMKit::NaiveBayes::GaussianNB.new }
  let(:rfr) { SVMKit::Ensemble::RandomForestRegressor.new(random_seed: 1) }
  let(:mae) { SVMKit::EvaluationMeasure::MeanAbsoluteError.new }

  it 'searches the best parameter among array-type parameters.' do
    param_grid = { scl__feature_range: [[-1.0, 1.0], [0.0, 1.0]] }
    pipe = SVMKit::Pipeline::Pipeline.new(steps: { scl: scl, nbs: nbs })
    gs = described_class.new(estimator: pipe, param_grid: param_grid, splitter: skfold)
    gs.fit(x, y)

    expect(gs.cv_results[:params].size).to eq(2)
    expect(gs.best_params[:scl__feature_range]).to be_a(Array)

    n_samples, = x.shape
    classes = y.to_a.uniq.sort
    probs = gs.predict_proba(x)
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(probs.shape).to match([n_samples, 3])
    expect(predicted).to eq(y)

    log_probs = gs.predict_log_proba(x)
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[log_probs[n, true].max_index] })]
    expect(log_probs.shape).to match([n_samples, 3])
    expect(predicted).to eq(y)
  end

  it 'searches the best parameter of pipeline.' do
    param_grid = {
      rbf__gamma: [32.0, 1.0],
      rbf__n_components: [4, 128],
      svc__reg_param: [16.0, 0.1]
    }
    pipe = SVMKit::Pipeline::Pipeline.new(steps: { rbf: rbf, svc: svc })
    gs = described_class.new(estimator: pipe, param_grid: param_grid, splitter: skfold)
    gs.fit(x_xor, y_xor)

    expect(gs.cv_results[:params].size).to eq(8)
    expect(gs.cv_results[:mean_test_score]).to be_a(Array)
    expect(gs.cv_results[:mean_test_score].size).to eq(8)
    expect(gs.cv_results[:std_test_score]).to be_a(Array)
    expect(gs.cv_results[:std_test_score].size).to eq(8)
    expect(gs.cv_results[:mean_train_score]).to be_a(Array)
    expect(gs.cv_results[:mean_train_score].size).to eq(8)
    expect(gs.cv_results[:std_train_score]).to be_a(Array)
    expect(gs.cv_results[:std_train_score].size).to eq(8)
    expect(gs.cv_results[:mean_fit_time]).to be_a(Array)
    expect(gs.cv_results[:mean_fit_time].size).to eq(8)
    expect(gs.cv_results[:std_fit_time]).to be_a(Array)
    expect(gs.cv_results[:std_fit_time].size).to eq(8)
    expect(gs.best_params).to be_a(Hash)
    expect(gs.best_params[:rbf__gamma]).to eq(1.0)
    expect(gs.best_params[:rbf__n_components]).to eq(128)
    expect(gs.best_params[:svc__reg_param]).to eq(0.1)
    expect(gs.best_estimator).to be_a(SVMKit::Pipeline::Pipeline)
    expect(gs.best_estimator.steps[:rbf].params[:gamma]).to eq(1.0)
    expect(gs.best_estimator.steps[:rbf].params[:n_components]).to eq(128)
    expect(gs.best_estimator.steps[:svc].params[:reg_param]).to eq(0.1)
    expect(gs.best_score).to eq(gs.cv_results[:mean_test_score].max)
    expect(gs.best_index).to eq(gs.cv_results[:mean_test_score].index(gs.cv_results[:mean_test_score].max))
    expect(gs.score(x_xor, y_xor)).to eq(gs.best_estimator.score(x_xor, y_xor))
    expect(gs.predict(x_xor)).to eq(gs.best_estimator.predict(x_xor))
    expect(gs.decision_function(x_xor)).to eq(gs.best_estimator.decision_function(x_xor))
  end

  it 'searches the best parameter of regressor.' do
    param_grid = { n_estimator: [1, 10], max_features: [1, 2] }
    gs = described_class.new(estimator: rfr, param_grid: param_grid, splitter: kfold,
                             evaluator: mae, greater_is_better: false)
    gs.fit(x_reg, y_reg)

    expect(gs.best_score).to eq(gs.cv_results[:mean_test_score].min)
    expect(gs.best_index).to eq(gs.cv_results[:mean_test_score].index(gs.cv_results[:mean_test_score].min))
    expect(gs.best_params).to eq(n_estimator: 10, max_features: 2)
    expect(gs.best_estimator.params[:n_estimator]).to eq(10)
    expect(gs.best_estimator.params[:max_features]).to eq(2)
    expect(gs.score(x_reg, y_reg)).to be_within(0.01).of(1.0)
  end

  it 'raises TypeError given a invalid param grid.' do
    expect { described_class.new(estimator: svc, param_grid: nil, splitter: skfold) }.to raise_error(TypeError)
    expect { described_class.new(estimator: svc, param_grid: [0], splitter: skfold) }.to raise_error(TypeError)
    expect { described_class.new(estimator: svc, param_grid: { reg_param: 0 }, splitter: skfold) }.to raise_error(TypeError)
  end

  it 'dumps and restores itself using Marshal module.' do
    param_grid = { svc__reg_param: [100.0, 0.01], fit_bias: [true, false] }
    estimator = described_class.new(estimator: svc, param_grid: param_grid, splitter: skfold)
    estimator.fit(x, y)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(copied.class).to eq(estimator.class)
    expect(copied.params[:estimator].class).to eq(estimator.params[:estimator].class)
    expect(copied.params[:param_grid]).to eq(estimator.params[:param_grid])
    expect(copied.params[:splitter].class).to eq(estimator.params[:splitter].class)
    expect(copied.params[:evaluator]).to eq(estimator.params[:evaluator])
    expect(copied.params[:greater_is_better]).to eq(estimator.params[:greater_is_better])
    expect(copied.cv_results).to eq(estimator.cv_results)
    expect(copied.best_score).to eq(estimator.best_score)
    expect(copied.best_params).to eq(estimator.best_params)
    expect(copied.best_index).to eq(estimator.best_index)
    expect(copied.best_estimator.class).to eq(estimator.best_estimator.class)
    expect(copied.best_estimator.weight_vec).to eq(estimator.best_estimator.weight_vec)
  end
end
