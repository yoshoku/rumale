# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NeuralNetwork::Loss::SoftmaxCrossEntropy do
  let(:y) { Numo::DFloat[[1, 5], [8, 2]] }
  let(:t) { Numo::DFloat[[0, 1], [1, 0]] }
  let(:loss) { described_class.new.call(y, t)[0] }
  let(:dout) { described_class.new.call(y, t)[1] }
  let(:z) do
    exp = Numo::NMath.exp(y - Numo::DFloat[[5, 8]].transpose)
    exp / exp.sum(axis: 1).expand_dims(1)
  end

  it 'calculates softmax cross entropy', :aggregate_failures do
    expect(loss).to eq(-(t * Numo::NMath.log(z + 1e-8)).sum.fdiv(2))
    expect(dout).to eq((z - t) / 2)
  end
end
