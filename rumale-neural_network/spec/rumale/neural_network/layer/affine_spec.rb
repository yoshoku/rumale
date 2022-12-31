# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NeuralNetwork::Layer::Affine do
  let(:rng) { Random.new(1) }
  let(:x) { Numo::DFloat[[1, 2], [3, 4], [5, 6]] }
  let(:z) { Numo::DFloat[[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]] }
  let(:n_inputs) { x.shape[1] }
  let(:n_outputs) { z.shape[1] }
  let(:adam) { Rumale::NeuralNetwork::Optimizer::Adam.new }
  let(:affine) { described_class.new(n_inputs: n_inputs, n_outputs: n_outputs, optimizer: adam, rng: rng.dup) }
  let(:rand_mat) { 0.01 * Rumale::Utils.rand_normal([n_inputs, n_outputs], rng.dup) }
  let(:out) { affine.forward(x)[0] }
  let(:backprop) { affine.forward(x)[1] }
  let(:dout) { backprop.call(z) }

  it 'performs linear transform', :aggregate_failures do
    expect(out).to be_a(Numo::DFloat)
    expect(out.ndim).to eq(z.ndim)
    expect(out.shape).to eq(z.shape)
    expect(out).to eq(x.dot(rand_mat))
    expect(backprop).to be_a(Proc)
    expect(dout).to be_a(Numo::DFloat)
    expect(dout.ndim).to eq(x.ndim)
    expect(dout.shape).to eq(x.shape)
    expect(dout).to eq(z.dot(rand_mat.transpose))
  end
end
