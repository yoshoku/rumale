# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NeuralNetwork::Layer::Relu do
  let(:x) { Numo::DFloat[[0.7, -0.5], [0.2, -0.1], [-0.4, 0.8]] }
  let(:z) { Numo::DFloat[[0.7, 0], [0.2, 0], [0, 0.8]] }
  let(:relu) { described_class.new }
  let(:out) { relu.forward(x)[0] }
  let(:backprop) { relu.forward(x)[1] }
  let(:dout) { backprop.call(x) }

  it 'calculates rectified linear function', :aggregate_failures do
    expect(out.class).to eq(Numo::DFloat)
    expect(out.ndim).to eq(x.ndim)
    expect(out.shape).to eq(x.shape)
    expect(out).to eq(z)
    expect(backprop.class).to eq(Proc)
    expect(dout.class).to eq(Numo::DFloat)
    expect(dout.ndim).to eq(x.ndim)
    expect(dout.shape).to eq(x.shape)
    expect(dout).to eq(z)
  end
end
