# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::Purity do
  let(:ground_truth) { Numo::Int32[1, 1, 2, 2, 3, 3, 0, 0, 4, 4] }
  let(:predicted) { Numo::Int32[2, 1, 1, 2, 0, 3, 0, 0, 4, 4] }
  let(:purity) { described_class.new.score(ground_truth, predicted) }

  it 'calculates purity of clustering result', :aggregate_failures do
    expect(purity).to be_a(Float)
    expect(purity).to be_within(1e-4).of(0.7)
  end
end
