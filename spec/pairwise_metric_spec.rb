require 'spec_helper'

RSpec.describe SVMKit::PairwiseMetric do
  let(:n_features) { 3 }
  let(:n_samples_a) { 10 }
  let(:n_samples_b) { 5 }
  let(:samples_a) { NMatrix.rand([n_samples_a, n_features]) }
  let(:samples_b) { NMatrix.rand([n_samples_b, n_features]) }

  it 'calculates pairwise euclidean distances.' do
    dist_mat_bf = NMatrix.new(
      [n_samples_a, n_samples_b],
      [*0...n_samples_a].product([*0...n_samples_b]).map { |m, n| (samples_a.row(m) - samples_b.row(n)).norm2 }
    )
    dist_mat = described_class.euclidean_distance(samples_a, samples_b)
    expect(dist_mat.shape[0]).to eq(n_samples_a)
    expect(dist_mat.shape[1]).to eq(n_samples_b)
    expect(dist_mat).to be_within(1.0e-5).of(dist_mat_bf)
  end
end
