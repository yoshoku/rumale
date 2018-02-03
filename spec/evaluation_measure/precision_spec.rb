require 'spec_helper'

RSpec.describe SVMKit::EvaluationMeasure::Precision do
  let(:bin_ground_truth) { Numo::Int32[1, 1, 1, 1, 1, -1, -1, -1, -1, -1] }
  let(:bin_predicted) { Numo::Int32[-1, -1, 1, 1, 1, -1, -1, 1, 1, 1] }
  # let(:mult_ground_truth) { Numo::Int32[0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 3, 3] }
  # let(:mult_predicted) { Numo::Int32[0, 2, 1, 0, 0, 1, 2, 2, 0, 2, 2, 0] }
  let(:mult_ground_truth) { Numo::Int32[0, 1, 2, 0, 1, 2, 3, 3, 0, 0] }
  let(:mult_predicted) { Numo::Int32[0, 2, 1, 2, 1, 0, 3, 3, 0, 0] }

  it 'calculates average precision for binary classification task.' do
    evaluator = described_class.new(average: 'binary')
    precision = evaluator.score(bin_ground_truth, bin_predicted)
    expect(precision.class).to eq(Float)
    expect(precision).to eq(0.5)
  end

  it 'calculates macro-average precision for multilabel classification task.' do
    evaluator = described_class.new(average: 'macro')
    precision = evaluator.score(mult_ground_truth, mult_predicted)
    expect(precision.class).to eq(Float)
    expect(precision).to eq(0.5625)
  end

  it 'calculates micro-average precision for multilabel classification task.' do
    evaluator = described_class.new(average: 'micro')
    precision = evaluator.score(mult_ground_truth, mult_predicted)
    expect(precision.class).to eq(Float)
    expect(precision).to eq(0.6)
  end

  it 'returns nil given an invalid average parameter.' do
    evaluator = described_class.new(average: 'foo')
    precision = evaluator.score(mult_ground_truth, mult_predicted)
    expect(precision.class).to eq(NilClass)
    expect(precision).to be_nil
  end
end
