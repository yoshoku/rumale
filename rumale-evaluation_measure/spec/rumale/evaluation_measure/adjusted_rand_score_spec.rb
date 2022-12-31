# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::AdjustedRandScore do
  let(:evaluator) { described_class.new }
  let(:ground_truth) { Numo::Int32[0, 0, 0, 1, 1, 1] }
  let(:predicted_a) { Numo::Int32[0, 0, 1, 1, 2, 2] }
  let(:predicted_b) { Numo::Int32[1, 1, 0, 0, 2, 2] }

  it 'calculates adjuested rand score of clustering result', :aggregate_failures do
    expect(evaluator.score(ground_truth, predicted_a)).to be_a(Float)
    expect(evaluator.score(ground_truth, predicted_a)).to be_within(1e-4).of(0.2424)
    expect(evaluator.score(ground_truth, predicted_b)).to be_within(1e-4).of(0.2424)
  end

  it 'returns one on special cases', :aggregate_failures do
    expect(evaluator.score(Numo::Int32[0, 1, 2], Numo::Int32[3, 4, 5])).to eq(1.0)
    expect(evaluator.score(Numo::Int32[0, 0, 0], Numo::Int32[1, 1, 1])).to eq(1.0)
    expect(evaluator.score(Numo::Int32[], Numo::Int32[])).to eq(1.0)
  end
end
