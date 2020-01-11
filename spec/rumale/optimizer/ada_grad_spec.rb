# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Optimizer::AdaGrad do
  let(:optimizer) { described_class.new(learning_rate: 0.1) }

  it 'dumps and restores itself using Marshal module.' do
    optimizer.call(Numo::DFloat.new(3).rand, Numo::DFloat.new(3).rand)
    copied = Marshal.load(Marshal.dump(optimizer))
    expect(optimizer.class).to eq(copied.class)
    expect(optimizer.params).to eq(copied.params)
    expect(optimizer.instance_variable_get(:@moment)).to eq(copied.instance_variable_get(:@moment))
  end
end
