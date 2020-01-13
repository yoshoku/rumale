# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Loss::EpsilonInsensitive do
  let(:y) { Numo::DFloat[4, 1, 1, 3] }
  let(:t) { Numo::DFloat[4, 3, 2, 1] }
  let(:loss) { described_class.new(epsilon: 1).loss(y, t) }
  let(:dout) { described_class.new(epsilon: 1).dloss(y, t) }

  it 'calculates epsilon insensitive loss', :aggregate_failures do
    expect(loss).to eq(0.5)
    expect(dout).to eq(Numo::DFloat[0, -1, 0, 1])
  end
end
