# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Base::ClusterAnalyzer do
  let(:dummy_class) do
    Class.new { include Rumale::Base::ClusterAnalyzer }.new
  end

  describe '#fit_predict' do
    it 'raises NotImplementedError when the fit method is not implemented' do
      expect { dummy_class.fit_predict }.to raise_error(NotImplementedError)
    end
  end

  describe '#score' do
    let(:ground_truth) { Numo::Int32[1, 1, 2, 2, 3, 3, 0, 0, 4, 4] }
    let(:predicted) { Numo::Int32[2, 1, 1, 2, 0, 3, 0, 0, 4, 4] }
    let(:score) { dummy_class.score(Numo::DFloat.new(10, 2).rand, ground_truth) }

    before { allow(dummy_class).to receive(:fit_predict).and_return(predicted) }

    it 'calculates purity', :aggregate_failures do
      expect(score).to be_a(Float)
      expect(score).to be_within(1e-4).of(0.7)
    end
  end
end
