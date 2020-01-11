# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Optimizer::YellowFin do
  let(:optimizer) { described_class.new(learning_rate: 0.05, momentum: 0.8, decay: 0.995, window_width: 2) }

  it 'dumps and restores itself using Marshal module.' do
    10.times { optimizer.call(Numo::DFloat[1, 2, 3], Numo::DFloat[0.1, 0.2, 0.3]) }
    copied = Marshal.load(Marshal.dump(optimizer))
    expect(optimizer.class).to eq(copied.class)
    expect(optimizer.params).to eq(copied.params)
    expect(optimizer.instance_variable_get(:@smth_learning_rate)).to eq(copied.instance_variable_get(:@smth_learning_rate))
    expect(optimizer.instance_variable_get(:@smth_momentum)).to eq(copied.instance_variable_get(:@smth_momentum))
    expect(optimizer.instance_variable_get(:@grad_norms)).to eq(copied.instance_variable_get(:@grad_norms))
    expect(optimizer.instance_variable_get(:@grad_norm_min)).to eq(copied.instance_variable_get(:@grad_norm_min))
    expect(optimizer.instance_variable_get(:@grad_norm_min)).to eq(copied.instance_variable_get(:@grad_norm_min))
    expect(optimizer.instance_variable_get(:@grad_norm_max)).to eq(copied.instance_variable_get(:@grad_norm_max))
    expect(optimizer.instance_variable_get(:@grad_mean_sqr)).to eq(copied.instance_variable_get(:@grad_mean_sqr))
    expect(optimizer.instance_variable_get(:@grad_mean)).to eq(copied.instance_variable_get(:@grad_mean))
    expect(optimizer.instance_variable_get(:@grad_var)).to eq(copied.instance_variable_get(:@grad_var))
    expect(optimizer.instance_variable_get(:@grad_norm_mean)).to eq(copied.instance_variable_get(:@grad_norm_mean))
    expect(optimizer.instance_variable_get(:@curve_mean)).to eq(copied.instance_variable_get(:@curve_mean))
    expect(optimizer.instance_variable_get(:@distance_mean)).to eq(copied.instance_variable_get(:@distance_mean))
    expect(optimizer.instance_variable_get(:@update)).to eq(copied.instance_variable_get(:@update))
  end
end
