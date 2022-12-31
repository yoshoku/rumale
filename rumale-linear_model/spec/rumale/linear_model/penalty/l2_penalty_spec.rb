# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Penalty::L2Penalty do
  let(:reg_param) { 0.1 }
  let(:lr) { 0.1 }
  let(:weight) { Numo::DFloat[100, 200, 300] }
  let(:regularizer) { described_class.new(reg_param: reg_param) }
  let(:penalized) { regularizer.call(weight, lr) }

  it 'regularized with L2 penalty' do
    expect(penalized).to eq(Numo::DFloat[99, 198, 297])
  end
end
