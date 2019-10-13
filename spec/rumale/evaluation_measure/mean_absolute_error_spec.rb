# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::MeanAbsoluteError do
  let(:ground_truth) { Numo::DFloat[3.2, -0.2, 2.2, 6.8] }
  let(:estimated) { Numo::DFloat[2.9, -0.1, 2.3, 7.2] }
  let(:mult_ground_truth) { Numo::DFloat[[0.5, 1.9], [-0.7, 1.8], [7.9, -6.5]] }
  let(:mult_estimated) { Numo::DFloat[[0.4, 2.0], [-0.8, 2.0], [8.3, -5.9]] }
  let(:evaluator) { described_class.new }

  it 'calculates mean absolute error for single regression task.' do
    mae = evaluator.score(ground_truth, estimated)
    expect(mae.class).to eq(Float)
    expect(mae).to be_within(1e-4).of(0.225)
  end

  it 'calculates mean absolute error for multiple regression task.' do
    mae = evaluator.score(mult_ground_truth, mult_estimated)
    expect(mae.class).to eq(Float)
    expect(mae).to be_within(1e-4).of(0.250)
  end

  it 'raises ArgumentError when the arrays with different shapes are given.' do
    expect { evaluator.score(ground_truth, mult_estimated) }.to raise_error(ArgumentError)
  end
end
