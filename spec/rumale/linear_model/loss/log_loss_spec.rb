# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Loss::LogLoss do
  let(:y) { Numo::DFloat[1, -1,  1, -1] }
  let(:t) { Numo::DFloat[1,  1, -1, -1] }
  let(:loss) { described_class.new.loss(y, t) }
  let(:dout) { described_class.new.dloss(y, t) }

  it 'calculates mean squared error', :aggregate_failures do
    expect((loss - 0.813261).abs).to be < 1e-4
    expect((dout - Numo::DFloat[-0.268941, -0.731059, 0.731059, 0.268941]).abs.mean).to be < 1e-4
  end
end
