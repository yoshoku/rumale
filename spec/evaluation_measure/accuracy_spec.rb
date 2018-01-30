require 'spec_helper'

RSpec.describe SVMKit::EvaluationMeasure::Accuracy do
  let(:bin_ground_truth) { Numo::Int32[1, 1, 1, 1, 1, -1, -1, -1, -1, -1] }
  let(:bin_predicted) { Numo::Int32[-1, -1, 1, 1, 1, -1, -1, 1, 1, 1] }
  let(:mult_ground_truth) { Numo::Int32[1, 1, 2, 2, 3, 3, 0, 0, 4, 4] }
  let(:mult_predicted) { Numo::Int32[5, 1, 5, 2, 5, 3, 5, 0, 5, 4] }

  it 'calculates accuracy for binary classification task.' do
    evaluator = described_class.new
    accuracy = evaluator.score(bin_ground_truth, bin_predicted)
    expect(accuracy.class).to eq(Float)
    expect(accuracy).to eq(0.5)
  end

  it 'calculates accuracy for multilabel classification task.' do
    evaluator = described_class.new
    accuracy = evaluator.score(mult_ground_truth, mult_predicted)
    expect(accuracy.class).to eq(Float)
    expect(accuracy).to eq(0.5)
  end
end
