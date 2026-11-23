# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::FScore do
  let(:bin_ground_truth) { Numo::Int32[1, 1, 1, 1, -1, -1, -1, -1] }
  let(:bin_predicted) { Numo::Int32[-1, -1, 1, 1, -1, -1, 1, 1] }
  let(:mult_ground_truth) { Numo::Int32[0, 1, 2, 0, 1, 2, 3, 3, 0, 0] }
  let(:mult_predicted) { Numo::Int32[0, 2, 1, 2, 1, 0, 3, 3, 0, 0] }

  it 'calculates average F1-score for binary classification task', :aggregate_failures do
    evaluator = described_class.new(average: 'binary')
    f_score = evaluator.score(bin_ground_truth, bin_predicted)
    expect(f_score).to be_a(Float)
    expect(f_score).to be_within(1e-4).of(0.5)
  end

  it 'calculates macro-average F1-score for multilabel classification task', :aggregate_failures do
    evaluator = described_class.new(average: 'macro')
    f_score = evaluator.score(mult_ground_truth, mult_predicted)
    expect(f_score).to be_a(Float)
    expect(f_score).to be_within(1e-4).of(0.5625)
  end

  it 'calculates micro-average F1-score for multilabel classification task', :aggregate_failures do
    evaluator = described_class.new(average: 'micro')
    f_score = evaluator.score(mult_ground_truth, mult_predicted)
    expect(f_score).to be_a(Float)
    expect(f_score).to be_within(1e-4).of(0.6)
  end

  it 'calculates micro-average F1-score when the prediction has labels missing from the ground truth', :aggregate_failures do
    evaluator = described_class.new(average: 'micro')
    f_score = evaluator.score(Numo::Int32[0, 0, 1], Numo::Int32[0, 2, 1])
    expect(f_score).to be_a(Float)
    expect(f_score).to be_within(1e-4).of(0.6667)
  end

  it 'returns zero for micro-average F1-score when no prediction is correct' do
    evaluator = described_class.new(average: 'micro')
    expect(evaluator.score(Numo::Int32[0, 1], Numo::Int32[1, 0])).to be_zero
  end

  it 'returns nil given an invalid average parameter' do
    evaluator = described_class.new(average: 'foo')
    f_score = evaluator.score(mult_ground_truth, mult_predicted)
    expect(f_score).to be_nil
  end
end
