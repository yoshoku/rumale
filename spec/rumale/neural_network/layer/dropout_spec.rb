# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NeuralNetwork::Layer::Dropout do
  let(:rng) { Random.new(1) }
  let(:x) { Rumale::Utils.rand_uniform([10, 10], rng.dup) }
  let(:dropout) { described_class.new(rate: 0.6, rng: rng.dup) }
  let(:out) { dropout.forward(x)[0] }
  let(:backprop) { dropout.forward(x)[1] }
  let(:dout) { backprop.call(x) }

  it 'performs dropout units', :aggregate_failures do
    expect(out.class).to eq(Numo::DFloat)
    expect(out.ndim).to eq(x.ndim)
    expect(out.shape).to eq(x.shape)
    expect(out.eq(0).count).to be_within(5).of(60)
    expect(backprop.class).to eq(Proc)
    expect(dout.class).to eq(Numo::DFloat)
    expect(dout.ndim).to eq(x.ndim)
    expect(dout.shape).to eq(x.shape)
    expect(dout.eq(0).count).to be_within(5).of(60)
  end
end
