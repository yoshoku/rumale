require 'spec_helper'

RSpec.describe SVMKit::PairwiseMetric do
  let(:n_features) { 3 }
  let(:n_samples_a) { 10 }
  let(:n_samples_b) { 5 }
  let(:samples_a) { NMatrix.rand([n_samples_a, n_features]) }
  let(:samples_b) { NMatrix.rand([n_samples_b, n_features]) }
  let(:gamma) { 0.5 }

  it 'calculates the euclidean distance matrix between different datasets.' do
    dist_mat_bf = NMatrix.new(
      [n_samples_a, n_samples_b],
      [*0...n_samples_a].product([*0...n_samples_b]).map { |m, n| (samples_a.row(m) - samples_b.row(n)).norm2 }
    )
    dist_mat = described_class.euclidean_distance(samples_a, samples_b)
    expect(dist_mat.shape[0]).to eq(n_samples_a)
    expect(dist_mat.shape[1]).to eq(n_samples_b)
    expect(dist_mat).to be_within(1.0e-5).of(dist_mat_bf)
  end

  it 'calculates the euclidean distance matrix of dataset.' do
    dist_mat_bf = NMatrix.new(
      [n_samples_a, n_samples_a],
      [*0...n_samples_a].product([*0...n_samples_b]).map { |m, n| (samples_a.row(m) - samples_a.row(n)).norm2 }
    )
    dist_mat = described_class.euclidean_distance(samples_a)
    expect(dist_mat.shape[0]).to eq(n_samples_a)
    expect(dist_mat.shape[1]).to eq(n_samples_a)
    expect(dist_mat).to be_within(1.0e-5).of(dist_mat_bf)
  end

  it 'calculates the RBF kernel matrix of dataset.' do
    kernel_mat_bf = NMatrix.new(
      [n_samples_a, n_samples_a],
      [*0...n_samples_a].product([*0...n_samples_b]).map do |m, n|
        dist = (samples_a.row(m) - samples_a.row(n)).norm2
        Math.exp(-gamma * (dist**2))
      end
    )
    kernel_mat = described_class.rbf_kernel(samples_a, nil, gamma)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_a)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end

  it 'calculates the RBF kernel matrix between different datasets.' do
    kernel_mat_bf = NMatrix.new(
      [n_samples_a, n_samples_b],
      [*0...n_samples_a].product([*0...n_samples_b]).map do |m, n|
        dist = (samples_a.row(m) - samples_b.row(n)).norm2
        Math.exp(-gamma * (dist**2))
      end
    )
    kernel_mat = described_class.rbf_kernel(samples_a, samples_b, gamma)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_b)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end
end
