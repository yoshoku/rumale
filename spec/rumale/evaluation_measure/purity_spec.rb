# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::Purity do
  let(:ground_truth) { Numo::Int32[1, 1, 2, 2, 3, 3, 0, 0, 4, 4] }
  let(:predicted) { Numo::Int32[2, 1, 1, 2, 0, 3, 0, 0, 4, 4] }

  it 'calculates purity of clustering result.' do
    evaluator = described_class.new
    purity = evaluator.score(ground_truth, predicted)
    expect(purity.class).to eq(Float)
    expect(purity).to eq(0.7)
  end
end
