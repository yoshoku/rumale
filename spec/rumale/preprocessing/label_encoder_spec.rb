# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::LabelEncoder do
  let(:encoder) { described_class.new }
  let(:labels) { Numo::Int32[0, 0, 2, 3, 2, 1] }
  let(:int_labels) { [1, 8, 8, 15, -1] }
  let(:str_labels) { %w[paris paris tokyo amsterdam] }

  it 'encodes labels.' do
    expect(encoder.fit_transform(labels)).to eq(labels)
  end

  it 'encodes integer labels, then decodes the encoded labels.' do
    encoded_labels = Numo::Int32[1, 2, 2, 3, 0]
    expect(encoder.fit_transform(int_labels)).to eq(encoded_labels)
    expect(encoder.inverse_transform(encoded_labels)).to eq(int_labels)
  end

  it 'encodes string labels, the decodes the encoded labels.' do
    encoded_labels = Numo::Int32[1, 1, 2, 0]
    expect(encoder.fit_transform(str_labels)).to eq(encoded_labels)
    expect(encoder.inverse_transform(encoded_labels)).to eq(str_labels)
  end

  it 'dumps and restores itself using Marshal module.' do
    encoder.fit(labels)
    copied = Marshal.load(Marshal.dump(encoder))
    expect(encoder.params).to eq(copied.params)
    expect(encoder.classes).to eq(copied.classes)
    expect(encoder.transform(labels)).to eq(copied.transform(labels))
  end
end
