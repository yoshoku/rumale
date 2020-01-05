# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Loss::MeanSquaredError do
  let(:y) { Numo::DFloat[1, 2, 3, 4] }
  let(:t) { Numo::DFloat[4, 3, 2, 1] }
  let(:loss) { described_class.new.loss(y, t) }
  let(:dout) { described_class.new.dloss(y, t) }

  it 'calculates mean squared error', :aggregate_failures do
    expect(loss).to eq(5)
    expect(dout).to eq(Numo::DFloat[-1.5, -0.5, 0.5, 1.5])
  end
end
