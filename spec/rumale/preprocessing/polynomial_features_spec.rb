# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::PolynomialFeatures do
  let(:x) { Numo::DFloat[[0, 1], [2, 3], [4, 5]] }
  let(:transformer) { described_class.new(degree: degree) }
  let(:z) { transformer.fit_transform(x) }

  context 'when degree is 0' do
    let(:degree) { 0 }

    it 'raises ArgumentError' do
      expect { transformer }.to raise_error(ArgumentError)
    end
  end

  context 'when degree is 1' do
    let(:degree) { 1 }

    it 'obtains polynomial expanded features', :aggregate_failures do
      expect(z).to eq(Numo::DFloat[
        [1, 0, 1],
        [1, 2, 3],
        [1, 4, 5]
      ])
      expect(transformer.n_output_features).to eq(3)
    end
  end

  context 'when degree is 2' do
    let(:degree) { 2 }

    it 'obtains polynomial expanded features', :aggregate_failures do
      expect(z).to eq(Numo::DFloat[
        [1, 0, 1, 0, 0, 1],
        [1, 2, 3, 4, 6, 9],
        [1, 4, 5, 16, 20, 25]
      ])
      expect(transformer.n_output_features).to eq(6)
    end
  end

  context 'when degree is 3' do
    let(:degree) { 3 }

    it 'obtains polynomial expanded features', :aggregate_failures do
      expect(z).to eq(Numo::DFloat[
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [1, 2, 3, 4, 6, 9, 8, 12, 18, 27],
        [1, 4, 5, 16, 20, 25, 64, 80, 100, 125]
      ])
      expect(transformer.n_output_features).to eq(10)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      transformer.fit(x)
      copied = Marshal.load(Marshal.dump(transformer))
      expect(copied.class).to eq(transformer.class)
      expect(copied.n_output_features).to eq(transformer.n_output_features)
    end
  end
end
