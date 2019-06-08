# frozen_string_literal: true

require 'spec_helper'

# rubocop:disable Style/WordArray, Style/SymbolArray
RSpec.describe Rumale::Preprocessing::OrdinalEncoder do
  let(:training) do
    x = [['left', 10, :a],
         ['right', 15, :b],
         ['right', 20, :c],
         ['up', 10, :a],
         ['up', 10, :c],
         ['down', 15, :b],
         ['right', 20, :c]]
    Numo::NArray.asarray(x)
  end
  let(:encoded_training) do
    x = [[1, 0, 0],
         [2, 1, 1],
         [2, 2, 2],
         [3, 0, 0],
         [3, 0, 2],
         [0, 1, 1],
         [2, 2, 2]]
    Numo::DFloat.asarray(x)
  end
  let(:encoded_training_b) do
    x = [[2, 1, 2],
         [3, 0, 0],
         [3, 2, 1],
         [0, 1, 2],
         [0, 1, 1],
         [1, 0, 0],
         [3, 2, 1]]
    Numo::NArray.asarray(x)
  end
  let(:testing) do
    x = [['up', 10, :c],
         ['down', 15, :b],
         ['right', 20, :a]]
    Numo::NArray.asarray(x)
  end
  let(:encoded_testing) do
    x = [[3, 0, 2],
         [0, 1, 1],
         [2, 2, 0]]
    Numo::DFloat.asarray(x)
  end
  let(:categories) { [['up', 'down', 'left', 'right'], [15, 10, 20], [:b, :c, :a]] }
  let(:encoder_auto) { described_class.new }
  let(:encoder) { described_class.new(categories: categories) }

  it 'encodes categorical features and decodes value features.' do
    encoded = encoder_auto.fit_transform(training)
    decoded = encoder_auto.inverse_transform(encoded)
    expect(encoder_auto.categories).to eq([['down', 'left', 'right', 'up'], [10, 15, 20], [:a, :b, :c]])
    expect(encoded.class).to eq(Numo::DFloat)
    expect(encoded.shape).to eq(training.shape)
    expect(encoded).to eq(encoded_training)
    expect(decoded.class).to eq(Numo::RObject)
    expect(decoded.shape).to eq(training.shape)
    expect(decoded.to_a).to eq(training.to_a)
  end

  it 'encodes testing data.' do
    encoder_auto.fit(training)
    encoded = encoder_auto.transform(testing)
    expect(encoded.class).to eq(Numo::DFloat)
    expect(encoded.shape).to eq(testing.shape)
    expect(encoded).to eq(encoded_testing)
  end

  it 'encodes with specified categories.' do
    encoded = encoder.transform(training)
    expect(encoder.categories).to eq(categories)
    expect(encoded.class).to eq(Numo::DFloat)
    expect(encoded.shape).to eq(training.shape)
    expect(encoded).to eq(encoded_training_b)
  end

  it 'dumps and restores itself using Marshal module.' do
    encoder.fit(training)
    copied = Marshal.load(Marshal.dump(encoder))
    expect(encoder.categories).to eq(copied.categories)
  end
end
# rubocop:enable Style/WordArray, Style/SymbolArray
