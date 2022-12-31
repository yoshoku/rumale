# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::NormalizedMutualInformation do
  let(:ground_truth) { Numo::Int32[1, 1, 2, 2, 3, 3, 0, 0, 4, 4] }
  let(:predicted) { Numo::Int32[0, 1, 1, 2, 0, 3, 0, 0, 4, 4] }

  it 'calculates normalized mutual information of clustering result', :aggregate_failures do
    evaluator = described_class.new
    nmi = evaluator.score(ground_truth, predicted)
    expect(nmi).to be_a(Float)
    expect(nmi).to be_within(5.0e-6).of(0.685653)
    expect(evaluator.score(Numo::Int32[1, 1, 0, 0], Numo::Int32[0, 0, 1, 1])).to eq(1.0)
    expect(evaluator.score(Numo::Int32[0, 0, 0, 0], Numo::Int32[0, 1, 2, 3])).to be_zero
  end
end
