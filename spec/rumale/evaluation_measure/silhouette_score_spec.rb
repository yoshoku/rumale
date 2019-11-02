# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::EvaluationMeasure::SilhouetteScore do
  let(:metric) { 'euclidean' }
  let(:score) do
    if metric == 'euclidean'
      described_class.new(metric: 'euclidean').score(x, y)
    else
      described_class.new(metric: 'precomputed').score(Rumale::PairwiseMetric.euclidean_distance(x), y)
    end
  end

  shared_examples 'calculates score' do
    context 'when all centroids are the same' do
      let(:x) { Numo::DFloat[[13, 4], [3, 2], [8, 5], [8, 1]] }
      let(:y) { Numo::Int32[1, 1, 2, 2] }

      it { expect(score).to be_within(0.001).of(-0.098) }
    end

    context 'when all samples are the same' do
      let(:x) { Numo::DFloat[[3, 4], [3, 4], [3, 4], [3, 4]] }
      let(:y) { Numo::Int32[1, 1, 2, 2] }

      it { expect(score).to eq(0.0) }
    end

    context 'when cluster have one sample' do
      let(:x) { Numo::DFloat[[0, 0], [2, 2], [3, 3], [5, 5]] }
      let(:y) { Numo::Int32[0, 0, 1, 2] }

      it { expect(score).to be_within(0.0001).of(-0.0416) }
    end
  end

  context "when metric is 'euclidean'" do
    let(:metric) { 'euclidean' }

    it_behaves_like 'calculates score'
  end

  context "when metric is 'precomputed'" do
    let(:metric) { 'precomputed' }

    it_behaves_like 'calculates score'
  end
end
