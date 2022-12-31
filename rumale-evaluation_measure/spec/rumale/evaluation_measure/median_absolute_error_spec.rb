# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::MedianAbsoluteError do
  let(:ground_truth) { Numo::DFloat[3, -0.5, 2, 7] }
  let(:estimated) { Numo::DFloat[2.5, 0.0, 2, 8] }
  let(:mae) { described_class.new.score(ground_truth, estimated) }

  it 'calculates median absolute error for single regression task', :aggregate_failures do
    expect(mae).to be_a(Float)
    expect(mae).to be_within(1e-4).of(0.5)
  end
end
