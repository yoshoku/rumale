# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::GridSearchCV do
  let(:three_clusters) { three_clusters_dataset }
  let(:x) { three_clusters[0] }
  let(:y) { three_clusters[1] }
  let(:xor) { xor_dataset }
  let(:x_xor) { xor[0] }
  let(:y_xor) { xor[1] }
  let(:x_reg) { two_clusters_dataset[0] }
  let(:y_reg) { x_reg[true, 0] + x_reg[true, 1]**2 }
  let(:kfold) { Rumale::ModelSelection::KFold.new(n_splits: 5, shuffle: true, random_seed: 1) }
  let(:skfold) { Rumale::ModelSelection::StratifiedKFold.new(n_splits: 5, shuffle: true, random_seed: 1) }
  let(:rbf) { Rumale::KernelApproximation::RBF.new(gamma: 0.1, random_seed: 1) }
  let(:lgt) { Rumale::LinearModel::LogisticRegression.new(random_seed: 1) }
  let(:scl) { Rumale::Preprocessing::MinMaxScaler.new }
  let(:nbs) { Rumale::NaiveBayes::GaussianNB.new }
  let(:dtr) { Rumale::Tree::DecisionTreeRegressor.new(random_seed: 1) }
  let(:mae) { Rumale::EvaluationMeasure::MeanAbsoluteError.new }

  it 'searches the best parameter among array-type parameters', :aggregate_failures do
    param_grid = { scl__feature_range: [[-1.0, 1.0], [0.0, 1.0]] }
    pipe = Rumale::Pipeline::Pipeline.new(steps: { scl: scl, nbs: nbs })
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

  it 'searches the best parameter of pipeline', :aggregate_failures do
    param_grid = {
      rbf__gamma: [32.0, 1.0],
      rbf__n_components: [4, 128],
      lgt__reg_param: [16.0, 0.1]
    }
    pipe = Rumale::Pipeline::Pipeline.new(steps: { rbf: rbf, lgt: lgt })
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
    expect(gs.best_params[:lgt__reg_param]).to eq(0.1)
    expect(gs.best_estimator).to be_a(Rumale::Pipeline::Pipeline)
    expect(gs.best_estimator.steps[:rbf].params[:gamma]).to eq(1.0)
    expect(gs.best_estimator.steps[:rbf].params[:n_components]).to eq(128)
    expect(gs.best_estimator.steps[:lgt].params[:reg_param]).to eq(0.1)
    expect(gs.best_score).to eq(gs.cv_results[:mean_test_score].max)
    expect(gs.best_index).to eq(gs.cv_results[:mean_test_score].index(gs.cv_results[:mean_test_score].max))
    expect(gs.score(x_xor, y_xor)).to eq(gs.best_estimator.score(x_xor, y_xor))
    expect(gs.predict(x_xor)).to eq(gs.best_estimator.predict(x_xor))
    expect(gs.decision_function(x_xor)).to eq(gs.best_estimator.decision_function(x_xor))
  end

  it 'searches the best parameter of regressor', :aggregate_failures do
    param_grid = { max_depth: [2, 4, 8], max_features: [1, 2] }
    gs = described_class.new(estimator: dtr, param_grid: param_grid, splitter: kfold,
                             evaluator: mae, greater_is_better: false)
    gs.fit(x_reg, y_reg)
    expect(gs.best_score).to eq(gs.cv_results[:mean_test_score].min)
    expect(gs.best_index).to eq(gs.cv_results[:mean_test_score].index(gs.cv_results[:mean_test_score].min))
    expect(gs.best_params).to eq(max_depth: 8, max_features: 2)
    expect(gs.best_estimator.params[:max_depth]).to eq(8)
    expect(gs.best_estimator.params[:max_features]).to eq(2)
    expect(gs.score(x_reg, y_reg)).to be_within(0.01).of(1.0)
  end

  it 'raises TypeError given a invalid param grid', :aggregate_failures do
    expect { described_class.new(estimator: lgt, param_grid: nil, splitter: skfold) }.to raise_error(TypeError)
    expect { described_class.new(estimator: lgt, param_grid: [0], splitter: skfold) }.to raise_error(TypeError)
    expect do
      described_class.new(estimator: lgt, param_grid: { reg_param: 0 }, splitter: skfold)
    end.to raise_error(TypeError)
  end
end
