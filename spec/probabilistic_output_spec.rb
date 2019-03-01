# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ProbabilisticOutput do
  let(:x_bin) { Marshal.load(File.read(__dir__ + '/test_samples.dat')) }
  let(:y_bin) { Marshal.load(File.read(__dir__ + '/test_labels.dat')) }
  let(:estimator) { Rumale::LinearModel::SVC.new(random_seed: 1) }

  it 'calculates class probability with svc' do
    n_samples, = x_bin.shape
    estimator.fit(x_bin, y_bin)
    df = estimator.decision_function(x_bin)
    prob_param = described_class.fit_sigmoid(df, y_bin)
    probs = Numo::DFloat.zeros(n_samples, 2)
    probs[true, 1] = 1 / (1 + Numo::NMath.exp(prob_param[0] * df + prob_param[1]))
    probs[true, 0] = 1 - probs[true, 1]
    predicted = Numo::Int32.cast(probs[true, 0] < probs[true, 1]) * 2 - 1
    expect(predicted).to eq(y_bin)
  end
end
