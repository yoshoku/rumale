# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Optimizer::SGD do
  let(:optimizer) { described_class.new(learning_rate: 0.1, momentum: 0.9, decay: 0.1) }

  it 'dumps and restores itself using Marshal module.' do
    optimizer.call(Numo::DFloat.new(3).rand, Numo::DFloat.new(3).rand)
    copied = Marshal.load(Marshal.dump(optimizer))
    expect(optimizer.class).to eq(copied.class)
    expect(optimizer.params).to eq(copied.params)
    expect(optimizer.instance_variable_get(:@update)).to eq(copied.instance_variable_get(:@update))
    expect(optimizer.instance_variable_get(:@iter)).to eq(copied.instance_variable_get(:@iter))
  end
end
