# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::LinearModel::Optimizer::SGD do
  let(:lr) { 0.2 }
  let(:momentum) { 0.5 }
  let(:decay) { 0.6 }
  let(:optimizer) { described_class.new(learning_rate: lr, momentum: momentum, decay: decay) }
  let(:weight) { Numo::DFloat[10, 20, 30] }
  let(:gradient) { Numo::DFloat[10, 20, 30] }

  it 'updates weight', :aggregate_failures do
    expect(optimizer.current_learning_rate).to eq(0.2)
    expect(optimizer.call(weight, gradient)).to eq(Numo::DFloat[8, 16, 24])
    expect(optimizer.current_learning_rate).to eq(0.125)
    expect(optimizer.call(weight, gradient)).to eq(Numo::DFloat[7.75, 15.5, 23.25])
  end
end
