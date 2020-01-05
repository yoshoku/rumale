# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Penalty::L1Penalty do
  let(:reg_param) { 100.0 }
  let(:lr) { 0.1 }
  let(:weight) { Numo::DFloat[10, -20, 30] }
  let(:regularizer) { described_class.new(reg_param: reg_param) }
  let(:penalized) { regularizer.call(weight, lr) }

  it 'regularized with L1 penalty', :aggregate_failures do
    expect(penalized).to eq(Numo::DFloat[0, -10, 20])
    expect(regularizer.call(penalized, lr)).to eq(Numo::DFloat[0, 0, 10])
  end
end
