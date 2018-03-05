# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::EvaluationMeasure::Recall do
  let(:bin_ground_truth) { Numo::Int32[1, 1, 1, 1, 1, -1, -1, -1, -1, -1] }
  let(:bin_predicted) { Numo::Int32[-1, -1, 1, 1, 1, -1, -1, 1, 1, 1] }
  let(:mult_ground_truth) { Numo::Int32[0, 1, 2, 0, 1, 2, 3, 3, 0, 0] }
  let(:mult_predicted) { Numo::Int32[0, 2, 1, 2, 1, 0, 3, 3, 0, 0] }

  it 'calculates average recall for binary classification task.' do
    evaluator = described_class.new(average: 'binary')
    recall = evaluator.score(bin_ground_truth, bin_predicted)
    expect(recall.class).to eq(Float)
    expect(recall).to eq(0.6)
  end

  it 'calculates macro-average recall for multilabel classification task.' do
    evaluator = described_class.new(average: 'macro')
    recall = evaluator.score(mult_ground_truth, mult_predicted)
    expect(recall.class).to eq(Float)
    expect(recall).to eq(0.5625)
  end

  it 'calculates micro-average recall for multilabel classification task.' do
    evaluator = described_class.new(average: 'micro')
    recall = evaluator.score(mult_ground_truth, mult_predicted)
    expect(recall.class).to eq(Float)
    expect(recall).to eq(0.6)
  end

  it 'returns nil given an invalid average parameter.' do
    evaluator = described_class.new(average: 'foo')
    recall = evaluator.score(mult_ground_truth, mult_predicted)
    expect(recall.class).to eq(NilClass)
    expect(recall).to be_nil
  end
end
