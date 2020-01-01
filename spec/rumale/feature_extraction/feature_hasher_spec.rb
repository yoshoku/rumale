# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::FeatureExtraction::FeatureHasher do
  let(:n_features) { 10 }
  let(:alternate_sign) { true }
  let(:encoder) { described_class.new(n_features: n_features, alternate_sign: alternate_sign) }
  let(:z) { encoder.fit_transform(x) }

  context 'when the samples consisting of value features' do
    let(:x) do
      [
        { dog: 1, cat: 2, elephant: 4 },
        { dog: 2, run: 5 }
      ]
    end

    it 'encodes to sample array' do
      expect(z).to eq(Numo::DFloat[[0, 0, -4, -1, 0, 0, 0, 0, 0, 2], [0, 0, 0, -2, -5, 0, 0, 0, 0, 0]])
    end
  end

  context 'when the samples containing of categorical features' do
    let(:n_features) { 4 }
    let(:x) do
      [
        { city: 'Dubai',  temperature: 33 },
        { city: 'London', temperature: 12 },
        { city: 'San Francisco', temperature: 18 }
      ]
    end

    it 'encodes to sample array' do
      expect(z).to eq(Numo::DFloat[[0, -33, -1, 0], [0, -12, -1, 0], [-1, -18, 0, 0]])
    end
  end
end
