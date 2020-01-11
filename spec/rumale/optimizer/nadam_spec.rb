# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Optimizer::Nadam do
  let(:optimizer) { described_class.new(learning_rate: 0.1, decay1: 0.8, decay2: 0.8) }

  it 'dumps and restores itself using Marshal module.' do
    optimizer.call(Numo::DFloat.new(3).rand, Numo::DFloat.new(3).rand)
    copied = Marshal.load(Marshal.dump(optimizer))
    expect(optimizer.class).to eq(copied.class)
    expect(optimizer.params).to eq(copied.params)
    expect(optimizer.instance_variable_get(:@fst_moment)).to eq(copied.instance_variable_get(:@fst_moment))
    expect(optimizer.instance_variable_get(:@sec_moment)).to eq(copied.instance_variable_get(:@sec_moment))
    expect(optimizer.instance_variable_get(:@decay1_prod)).to eq(copied.instance_variable_get(:@decay1_prod))
    expect(optimizer.instance_variable_get(:@iter)).to eq(copied.instance_variable_get(:@iter))
  end
end
