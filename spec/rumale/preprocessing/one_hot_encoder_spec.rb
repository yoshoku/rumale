# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::OneHotEncoder do
  let(:encoder) { described_class.new }
  let(:labels) { Numo::Int32[0, 0, 2, 3, 2, 1] }
  let(:codes) { Numo::DFloat[[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]] }

  it 'encodes multi-label vector into one-hot-vectors' do
    expect(encoder.fit_transform(labels)).to eq(codes)
  end

  it 'encodes samples into one-hot-vectors categorically' do
    x = Numo::Int32[[0, 0, 10], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
    y = Numo::Int32[[0, 1, 1]]
    encoder.fit(x)
    expect(encoder.n_values).to eq(Numo::Int32[2, 3, 11])
    expect(encoder.active_features).to eq(Numo::Int32[0, 1, 2, 3, 4, 5, 6, 7, 15])
    expect(encoder.feature_indices).to eq(Numo::Int32[0, 2, 5, 16])
    expect(encoder.transform(y)).to eq(Numo::DFloat[[1, 0, 0, 1, 0, 0, 1, 0, 0]])
  end

  it 'dumps and restores itself using Marshal module' do
    encoder.fit(labels)
    copied = Marshal.load(Marshal.dump(encoder))
    expect(encoder.params).to eq(copied.params)
    expect(encoder.n_values).to eq(copied.n_values)
    expect(encoder.active_features).to eq(copied.active_features)
    expect(encoder.feature_indices).to eq(copied.feature_indices)
    expect(encoder.transform(labels)).to eq(copied.transform(labels))
  end

  it 'raises ArgumentError when given the sample contains negative value' do
    x = Numo::Int32[[-1, 0, 1], [0, 1, 2]]
    expect { encoder.fit(x) }.to raise_error(ArgumentError, 'Expected the input samples only consists of non-negative integer values.')
    expect { encoder.transform(x) }.to raise_error(ArgumentError, 'Expected the input samples only consists of non-negative integer values.')
    expect { encoder.fit_transform(x) }.to raise_error(ArgumentError, 'Expected the input samples only consists of non-negative integer values.')
  end
end
