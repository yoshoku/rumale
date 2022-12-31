# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Base::Classifier do
  let(:dummy_class) do
    Class.new { include Rumale::Base::Classifier }.new
  end

  describe '#fit' do
    it 'raises NotImplementedError when the fit method is not implemented' do
      expect { dummy_class.fit }.to raise_error(NotImplementedError)
    end
  end

  describe '#predict' do
    it 'raises NotImplementedError when the predict method is not implemented' do
      expect { dummy_class.predict }.to raise_error(NotImplementedError)
    end
  end

  describe '#score' do
    let(:ground_truth) { Numo::Int32[1, 1, 1, 1, 1, -1, -1, -1, -1, -1] }
    let(:predicted) { Numo::Int32[-1, -1, 1, 1, 1, -1, -1, 1, 1, 1] }
    let(:score) { dummy_class.score(Numo::DFloat.new(10, 2).rand, ground_truth) }

    before { allow(dummy_class).to receive(:predict).and_return(predicted) }

    it 'calculates accuracy', :aggregate_failures do
      expect(score).to be_a(Float)
      expect(score).to be_within(1e-4).of(0.5)
    end
  end
end
