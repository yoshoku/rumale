# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::MeanSquaredLogError do
  let(:ground_truth) { Numo::DFloat[3, 5, 2.5, 7] }
  let(:estimated) { Numo::DFloat[2.5, 5, 4, 8] }
  let(:mult_ground_truth) { Numo::DFloat[[0.5, 1], [1, 2], [7, 6]] }
  let(:mult_estimated) { Numo::DFloat[[0.5, 2], [1, 2.5], [8, 8]] }
  let(:evaluator) { described_class.new }

  it 'calculates mean squared logarithmic error for single regression task', :aggregate_failures do
    msle = evaluator.score(ground_truth, estimated)
    expect(msle).to be_a(Float)
    expect(msle).to be_within(1e-4).of(0.0397)
  end

  it 'calculates mean squared logarithmic error for multiple regression task', :aggregate_failures do
    msle = evaluator.score(mult_ground_truth, mult_estimated)
    expect(msle).to be_a(Float)
    expect(msle).to be_within(1e-4).of(0.0441)
  end
end
