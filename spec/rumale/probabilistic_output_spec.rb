# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ProbabilisticOutput do
  let(:two_clusters) { two_clusters_dataset }
  let(:x_bin) { two_clusters[0] }
  let(:y_bin) { two_clusters[1] }
  let(:estimator) { Rumale::LinearModel::SVC.new(random_seed: 1) }

  it 'calculates class probability with svc' do
    classes = y_bin.to_a.uniq.sort
    n_samples, = x_bin.shape
    estimator.fit(x_bin, y_bin)
    df = estimator.decision_function(x_bin)
    prob_param = described_class.fit_sigmoid(df, y_bin)
    probs = Numo::DFloat.zeros(n_samples, 2)
    probs[true, 1] = 1 / (1 + Numo::NMath.exp(prob_param[0] * df + prob_param[1]))
    probs[true, 0] = 1 - probs[true, 1]
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(predicted).to eq(y_bin)
  end
end
