# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::EvaluationMeasure::R2Score do
  let(:ground_truth) { Numo::DFloat[3, -0.2, 2, 7] }
  let(:estimated)    { Numo::DFloat[2.5, 0.0, 2, 7.2] }
  let(:mult_ground_truth) { Numo::DFloat[[0.5, 1], [-0.7, 1], [7, -6]] }
  let(:mult_estimated)    { Numo::DFloat[[0.1, 2], [-0.8, 2], [8, -5]] }
  let(:evaluator) { described_class.new }

  it 'calculates average R^2-score for single regression task.' do
    r2_score = evaluator.score(ground_truth, estimated)
    expect(r2_score.class).to eq(Float)
    expect(r2_score).to be_within(1e-6).of(0.987881)
  end
  
  it 'calculates average R^2-score for multiple regression task.' do
    r2_score = evaluator.score(mult_ground_truth, mult_estimated)
    expect(r2_score.class).to eq(Float)
    expect(r2_score).to be_within(1e-6).of(0.937039)
  end

  it 'returns zero if the division by zero occurs in calculation process.' do
    r2_score = evaluator.score(Numo::DFloat[2, 2, 2], Numo::DFloat[1, 0.5, 1.5])
    expect(r2_score.class).to eq(Float)
    expect(r2_score).to be_zero
  end

  it 'raises ArgumentError when the arrays with different shapes are given.' do
    expect { evaluator.score(ground_truth, mult_estimated) }.to raise_error(ArgumentError) 
  end
end
