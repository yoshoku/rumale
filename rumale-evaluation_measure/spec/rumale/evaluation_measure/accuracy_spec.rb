# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::Accuracy do
  let(:accuracy) { described_class.new.score(ground_truth, predicted) }

  context 'when binary classification problem' do
    let(:ground_truth) { Numo::Int32[1, 1, 1, 1, 1, -1, -1, -1, -1, -1] }
    let(:predicted) { Numo::Int32[-1, -1, 1, 1, 1, -1, -1, 1, 1, 1] }

    it 'calculates accuracy for binary classification task', :aggregate_failures do
      expect(accuracy).to be_a(Float)
      expect(accuracy).to be_within(1e-4).of(0.5)
    end
  end

  context 'when multiclass classification problem' do
    let(:ground_truth) { Numo::Int32[1, 1, 2, 2, 3, 3, 0, 0, 4, 4] }
    let(:predicted) { Numo::Int32[5, 1, 5, 2, 5, 3, 5, 0, 5, 4] }

    it 'calculates accuracy for multilabel classification task', :aggregate_failures do
      expect(accuracy).to be_a(Float)
      expect(accuracy).to be_within(1e-4).of(0.5)
    end
  end
end
