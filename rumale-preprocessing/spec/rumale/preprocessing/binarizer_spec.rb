# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::Binarizer do
  let(:x) { Numo::DFloat[[-0.5, 15.2], [3.1, -2.4], [4.2, 12.6]] }
  let(:binarized) { binarizer.fit_transform(x) }
  let(:n_nonzero_elements) { binarized.gt(0).count }

  context 'when threshold value is not given' do
    let(:binarizer) { described_class.new }

    it 'binarizes given samples', :aggregate_failures do
      expect(binarized).to eq(Numo::DFloat[[0, 1], [1, 0], [1, 1]])
      expect(n_nonzero_elements).to eq(4)
    end
  end

  context 'when threshold value is given' do
    let(:threshold) { 5.0 }
    let(:binarizer) { described_class.new(threshold: threshold) }

    it 'binarizes given samples with given threshold value', :aggregate_failures do
      expect(binarized).to eq(Numo::DFloat[[0, 1], [0, 0], [0, 1]])
      expect(n_nonzero_elements).to eq(2)
    end
  end
end
