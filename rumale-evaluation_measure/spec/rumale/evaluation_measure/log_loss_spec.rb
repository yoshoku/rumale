# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::LogLoss do
  let(:bin_ground_truth) { Numo::Int32[1, 1, 1, 1, 1, -1, -1, -1, -1, -1] }
  let(:bin_predicted) { Numo::DFloat[0.9, 0.8, 0.6, 0.6, 0.8, 0.1, 0.2, 0.4, 0.4, 0.2] }
  let(:mult_ground_truth) { Numo::Int32[1, 0, 0, 2] }
  let(:mult_predicted) { Numo::DFloat[[0.3, 0.5, 0.2], [0.7, 0.3, 0.0], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]] }
  let(:evaluator) { described_class.new }

  it 'calculates logarithmic loss for binary classification task', :aggregate_failures do
    log_loss = evaluator.score(bin_ground_truth, bin_predicted)
    expect(log_loss).to be_a(Float)
    expect(log_loss).to be_within(1e-6).of(0.314659)
  end

  it 'calculates logarithmic loss for multilabel classification task', :aggregate_failures do
    log_loss = evaluator.score(mult_ground_truth, mult_predicted)
    expect(log_loss).to be_a(Float)
    expect(log_loss).to be_within(1e-6).of(0.407410)
  end
end
