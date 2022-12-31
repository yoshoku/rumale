# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ProbabilisticOutput do
  let(:n_samples) { 6 }
  let(:n_classes) { 2 }
  let(:df) { Numo::DFloat[-2.5, 1.2, -0.8, -1.3, 2.4, 0.8] }
  let(:y_bin) { Numo::Int32[0, 1, 0, 0, 1, 1] }
  let(:prob_params) { described_class.fit_sigmoid(df, y_bin) }
  let(:probs) { 1.0 / (1.0 + Numo::NMath.exp(prob_params[0] * df + prob_params[1])) }
  let(:predicted) { Numo::Int32.cast(0.5 * ((probs - 0.5).sign + 1)) }

  it 'calculates class probability', :aggregate_failures do
    expect(probs.to_a).to(be_all { |v| (0...1).cover?(v) })
    expect(predicted).to eq(y_bin)
  end
end
