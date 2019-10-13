# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::LabelBinarizer do
  let(:encoder) { described_class.new.fit(labels) }
  let(:labels) { Numo::Int32[0, -1, 3, 3, 1, 1] }
  let(:encoded_labels) { Numo::Int32[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]] }

  it 'encodes labels.' do
    encoder.fit_transform(labels)
    expect(encoder.transform(labels)).to eq(encoded_labels)
  end

  it 'decode binary labels.' do
    expect(encoder.inverse_transform(encoded_labels)).to eq(labels.to_a)
  end

  it 'dumps and restores itself using Marshal module.' do
    copied = Marshal.load(Marshal.dump(encoder))
    expect(encoder.params).to eq(copied.params)
    expect(encoder.classes).to eq(copied.classes)
    expect(encoder.transform(labels)).to eq(copied.transform(labels))
  end
end
