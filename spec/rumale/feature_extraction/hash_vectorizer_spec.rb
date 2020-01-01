# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::FeatureExtraction::HashVectorizer do
  let(:separator) { '=' }
  let(:sort) { true }
  let(:encoder) { described_class.new(separator: separator, sort: sort) }
  let(:z) { encoder.fit_transform(x) }
  let(:copied) { Marshal.load(Marshal.dump(encoder.fit(x))) }

  context 'when the samples consisting of value features' do
    let(:x) do
      [
        { foo: 1, bar: 2 },
        { foo: 3, baz: 1 }
      ]
    end

    it 'encodes sample matrix', :aggragate_failures do
      expect(z).to eq(Numo::DFloat[[2, 0, 1], [0, 1, 3]])
      expect(encoder.feature_names).to eq(%i[bar baz foo])
      expect(encoder.vocabulary).to eq(foo: 2, bar: 0, baz: 1)
      expect(encoder.inverse_transform(z)).to eq(x)
    end

    it 'encodes out of sample' do
      expect(encoder.fit(x).transform(foo: 4, unseen_feature: 3)).to eq(Numo::DFloat[[0, 0, 4]])
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(copied.feature_names).to eq(encoder.feature_names)
      expect(copied.vocabulary).to eq(encoder.vocabulary)
    end
  end

  context 'when the samples containing of categorical features' do
    let(:x) do
      [
        { city: 'Dubai',  temperature: 33 },
        { city: 'London', temperature: 12 },
        { city: 'San Francisco', temperature: 18 }
      ]
    end

    it 'encodes sample matrix', :aggragate_failures do
      expect(z).to eq(Numo::DFloat[[1, 0, 0, 33], [0, 1, 0, 12], [0, 0, 1, 18]])
      expect(encoder.feature_names).to eq([:'city=Dubai', :'city=London', :'city=San Francisco', :temperature])
      expect(encoder.vocabulary).to eq('city=Dubai': 0, 'city=London': 1, 'city=San Francisco': 2, temperature: 3)
      expect(encoder.inverse_transform(z)).to eq(x)
    end

    it 'encodes out of sample' do
      expect(encoder.fit(x).transform(city: 'Tokyo', temperature: 10)).to eq(Numo::DFloat[[0, 0, 0, 10]])
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(copied.feature_names).to eq(encoder.feature_names)
      expect(copied.vocabulary).to eq(encoder.vocabulary)
    end
  end
end
