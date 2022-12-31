# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::CalinskiHarabaszScore do
  let(:score) { described_class.new.score(x, y) }

  context 'when all centroids are the same' do
    let(:x) { Numo::DFloat[[13, 4], [3, 2], [8, 5], [8, 1]] }
    let(:y) { Numo::Int32[1, 1, 2, 2] }

    it { expect(score).to be_zero }
  end

  context 'when all samples are the same' do
    let(:x) { Numo::DFloat[[3, 4], [3, 4], [3, 4], [3, 4]] }
    let(:y) { Numo::Int32[1, 1, 2, 2] }

    it { expect(score).to eq(1.0) }
  end

  context 'when cluster have one sample' do
    let(:x) { Numo::DFloat[[0, 0], [2, 2], [3, 3], [5, 5]] }
    let(:y) { Numo::Int32[0, 0, 1, 2] }

    it { expect(score).to be_within(0.01).of(2.75) }
  end
end
