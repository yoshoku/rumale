# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Loss::HingeLoss do
  let(:y) { Numo::DFloat[1, -1, 1, -1] }
  let(:t) { Numo::DFloat[1, 1, -1, -1] }
  let(:loss) { described_class.new.loss(y, t) }
  let(:dout) { described_class.new.dloss(y, t) }

  it 'calculates hinge loss', :aggregate_failures do
    expect(loss).to eq(1)
    expect(dout).to eq(Numo::DFloat[0, -1, 1, 0])
  end
end
