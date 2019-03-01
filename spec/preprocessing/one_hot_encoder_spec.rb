# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::OneHotEncoder do
  let(:encoder) { described_class.new }
  let(:labels) { Numo::Int32[0, 0, 2, 3, 2, 1] }
  let(:codes) { Numo::DFloat[[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]] }

  it 'encodes multi-label vector into one-hot-vectors' do
    expect(encoder.fit_transform(labels)).to eq(codes)
  end

  it 'dumps and restores itself using Marshal module.' do
    encoder.fit(labels)
    copied = Marshal.load(Marshal.dump(encoder))
    expect(encoder.params).to eq(copied.params)
    expect(encoder.n_values).to eq(copied.n_values)
    expect(encoder.feature_indices).to eq(copied.feature_indices)
    expect(encoder.transform(labels)).to eq(copied.transform(labels))
  end

  # it 'encodes samples into one-hot-vectors like as the scikit-learn example' do
  #   x = Numo::Int32[[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
  #   y = Numo::Int32[[0, 1, 1]]
  #   expect(encoder.fit(x).transform(y)).to eq(Numo::Int32[[1, 0, 0, 1, 0, 0, 1, 0, 0]])
  # end
end
