# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::ExplainedVarianceScore do
  let(:ground_truth) { Numo::DFloat[3, -0.5, 2, 7] }
  let(:estimated) { Numo::DFloat[2.5, 0.0, 2, 8] }
  let(:mult_ground_truth) { Numo::DFloat[[0.5, 1], [-1, 1], [7, -6]] }
  let(:mult_estimated) { Numo::DFloat[[0, 2], [-1, 2], [8, -5]] }
  let(:evaluator) { described_class.new }

  it 'calculates explained variance score for single regression task', :aggregate_failures do
    evs = evaluator.score(ground_truth, estimated)
    expect(evs).to be_a(Float)
    expect(evs).to be_within(1e-4).of(0.9571)
  end

  it 'calculates explained variance score for multiple regression task', :aggregate_failures do
    evs = evaluator.score(mult_ground_truth, mult_estimated)
    expect(evs).to be_a(Float)
    expect(evs).to be_within(1e-4).of(0.9838)
  end
end
