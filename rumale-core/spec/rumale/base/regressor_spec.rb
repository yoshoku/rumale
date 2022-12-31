# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Base::Regressor do
  let(:dummy_class) do
    Class.new { include Rumale::Base::Regressor }.new
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
    let(:score) { dummy_class.score(Numo::DFloat.new(4, 2).rand, ground_truth) }

    before { allow(dummy_class).to receive(:predict).and_return(predicted) }

    context 'when single target regression' do
      let(:ground_truth) { Numo::DFloat[3, -0.2, 2, 7] }
      let(:predicted) { Numo::DFloat[2.5, 0.0, 2, 7.2] }

      it 'calculate R^2-score', :aggregate_failures do
        expect(score).to be_a(Float)
        expect(score).to be_within(1e-6).of(0.987881)
      end
    end

    context 'when multi-target regression' do
      let(:ground_truth) { Numo::DFloat[[0.5, 1], [-0.7, 1], [7, -6], [3.2, 4]] }
      let(:predicted) { Numo::DFloat[[0.1, 2], [-0.8, 2], [8, -5], [3, 3.8]] }

      it 'calculate R^2-score', :aggregate_failures do
        expect(score).to be_a(Float)
        expect(score).to be_within(1e-6).of(0.954556)
      end
    end
  end
end
