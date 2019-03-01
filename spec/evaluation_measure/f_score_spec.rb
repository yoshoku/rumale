# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::FScore do
  let(:bin_ground_truth) { Numo::Int32[1, 1, 1, 1, -1, -1, -1, -1] }
  let(:bin_predicted) { Numo::Int32[-1, -1, 1, 1, -1, -1, 1, 1] }
  let(:mult_ground_truth) { Numo::Int32[0, 1, 2, 0, 1, 2, 3, 3, 0, 0] }
  let(:mult_predicted) { Numo::Int32[0, 2, 1, 2, 1, 0, 3, 3, 0, 0] }

  it 'calculates average F1-score for binary classification task.' do
    evaluator = described_class.new(average: 'binary')
    f_score = evaluator.score(bin_ground_truth, bin_predicted)
    expect(f_score.class).to eq(Float)
    expect(f_score).to eq(0.5)
  end

  it 'calculates macro-average F1-score for multilabel classification task.' do
    evaluator = described_class.new(average: 'macro')
    f_score = evaluator.score(mult_ground_truth, mult_predicted)
    expect(f_score.class).to eq(Float)
    expect(f_score).to eq(0.5625)
  end

  it 'calculates micro-average F1-score for multilabel classification task.' do
    evaluator = described_class.new(average: 'micro')
    f_score = evaluator.score(mult_ground_truth, mult_predicted)
    expect(f_score.class).to eq(Float)
    expect(f_score).to eq(0.6)
  end

  it 'returns nil given an invalid average parameter.' do
    evaluator = described_class.new(average: 'foo')
    f_score = evaluator.score(mult_ground_truth, mult_predicted)
    expect(f_score.class).to eq(NilClass)
    expect(f_score).to be_nil
  end
end
