# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::PairwiseMetric do
  let(:n_features) { 3 }
  let(:n_samples_a) { 10 }
  let(:n_samples_b) { 5 }
  let(:samples_a) { Numo::DFloat.new(n_samples_a, n_features).rand }
  let(:samples_b) { Numo::DFloat.new(n_samples_b, n_features).rand }
  let(:degree) { 3 }
  let(:gamma) { 0.5 }
  let(:coef) { 1 }

  it 'calculates the pairwise squared errors between different datasets.' do
    dist_mat_bf = Numo::DFloat.zeros(n_samples_a, n_samples_b)
    n_samples_a.times do |m|
      n_samples_b.times do |n|
        dist_mat_bf[m, n] = ((samples_a[m, true] - samples_b[n, true])**2).sum
      end
    end
    dist_mat = described_class.squared_error(samples_a, samples_b)
    expect(dist_mat.class).to eq(Numo::DFloat)
    expect(dist_mat.shape[0]).to eq(n_samples_a)
    expect(dist_mat.shape[1]).to eq(n_samples_b)
    expect(dist_mat).to be_within(1.0e-5).of(dist_mat_bf)
  end

  it 'calculates the euclidean distance matrix between different datasets.' do
    dist_mat_bf = Numo::DFloat.zeros(n_samples_a, n_samples_b)
    n_samples_a.times do |m|
      n_samples_b.times do |n|
        dist_mat_bf[m, n] = Math.sqrt(((samples_a[m, true] - samples_b[n, true])**2).sum)
      end
    end
    dist_mat = described_class.euclidean_distance(samples_a, samples_b)
    expect(dist_mat.class).to eq(Numo::DFloat)
    expect(dist_mat.shape[0]).to eq(n_samples_a)
    expect(dist_mat.shape[1]).to eq(n_samples_b)
    expect(dist_mat).to be_within(1.0e-5).of(dist_mat_bf)
  end

  it 'calculates the euclidean distance matrix of dataset.' do
    dist_mat_bf = Numo::DFloat.zeros(n_samples_a, n_samples_a)
    n_samples_a.times do |m|
      n_samples_a.times do |n|
        dist_mat_bf[m, n] = Math.sqrt(((samples_a[m, true] - samples_a[n, true])**2).sum)
      end
    end
    dist_mat = described_class.euclidean_distance(samples_a)
    expect(dist_mat.class).to eq(Numo::DFloat)
    expect(dist_mat.shape[0]).to eq(n_samples_a)
    expect(dist_mat.shape[1]).to eq(n_samples_a)
    expect(dist_mat).to be_within(1.0e-5).of(dist_mat_bf)
  end

  it 'calculates the RBF kernel matrix of dataset.' do
    kernel_mat_bf = Numo::DFloat.asarray(
      [*0...n_samples_a].product([*0...n_samples_a]).map do |m, n|
        dist = Math.sqrt(((samples_a[m, true] - samples_a[n, true])**2).sum)
        Math.exp(-gamma * (dist**2))
      end
    ).reshape(n_samples_a, n_samples_a)
    kernel_mat = described_class.rbf_kernel(samples_a, nil, gamma)
    expect(kernel_mat.class).to eq(Numo::DFloat)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_a)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end

  it 'calculates the RBF kernel matrix between different datasets.' do
    kernel_mat_bf = Numo::DFloat.asarray(
      [*0...n_samples_a].product([*0...n_samples_b]).map do |m, n|
        dist = Math.sqrt(((samples_a[m, true] - samples_b[n, true])**2).sum)
        Math.exp(-gamma * (dist**2))
      end
    ).reshape(n_samples_a, n_samples_b)
    kernel_mat = described_class.rbf_kernel(samples_a, samples_b, gamma)
    expect(kernel_mat.class).to eq(Numo::DFloat)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_b)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end

  it 'calculates the linear kernel matrix of dataset.' do
    kernel_mat_bf = Numo::DFloat.asarray(
      [*0...n_samples_a].product([*0...n_samples_a]).map do |m, n|
        samples_a[m, true].dot(samples_a[n, true].transpose).to_f
      end
    ).reshape(n_samples_a, n_samples_a)
    kernel_mat = described_class.linear_kernel(samples_a, nil)
    expect(kernel_mat.class).to eq(Numo::DFloat)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_a)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end

  it 'calculates the linear kernel matrix between different datasets.' do
    kernel_mat_bf = Numo::DFloat.asarray(
      [*0...n_samples_a].product([*0...n_samples_b]).map do |m, n|
        samples_a[m, true].dot(samples_b[n, true].transpose).to_f
      end
    ).reshape(n_samples_a, n_samples_b)
    kernel_mat = described_class.linear_kernel(samples_a, samples_b)
    expect(kernel_mat.class).to eq(Numo::DFloat)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_b)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end

  it 'calculates the polynomial kernel matrix of dataset.' do
    kernel_mat_bf = Numo::DFloat.asarray(
      [*0...n_samples_a].product([*0...n_samples_a]).map do |m, n|
        (samples_a[m, true].dot(samples_a[n, true].transpose).to_f * gamma + coef)**degree
      end
    ).reshape(n_samples_a, n_samples_a)
    kernel_mat = described_class.polynomial_kernel(samples_a, nil, degree, gamma, coef)
    expect(kernel_mat.class).to eq(Numo::DFloat)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_a)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end

  it 'calculates the polynomial kernel matrix between different datasets.' do
    kernel_mat_bf = Numo::DFloat.asarray(
      [*0...n_samples_a].product([*0...n_samples_b]).map do |m, n|
        (samples_a[m, true].dot(samples_b[n, true].transpose).to_f * gamma + coef)**degree
      end
    ).reshape(n_samples_a, n_samples_b)
    kernel_mat = described_class.polynomial_kernel(samples_a, samples_b, degree, gamma, coef)
    expect(kernel_mat.class).to eq(Numo::DFloat)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_b)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end

  it 'calculates the sigmoid kernel matrix of dataset.' do
    kernel_mat_bf = Numo::DFloat.asarray(
      [*0...n_samples_a].product([*0...n_samples_a]).map do |m, n|
        Math.tanh(samples_a[m, true].dot(samples_a[n, true].transpose).to_f * gamma + coef)
      end
    ).reshape(n_samples_a, n_samples_a)
    kernel_mat = described_class.sigmoid_kernel(samples_a, nil, gamma, coef)
    expect(kernel_mat.class).to eq(Numo::DFloat)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_a)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end

  it 'calculates the sigmoid kernel matrix between different datasets.' do
    kernel_mat_bf = Numo::DFloat.asarray(
      [*0...n_samples_a].product([*0...n_samples_b]).map do |m, n|
        Math.tanh(samples_a[m, true].dot(samples_b[n, true].transpose).to_f * gamma + coef)
      end
    ).reshape(n_samples_a, n_samples_b)
    kernel_mat = described_class.sigmoid_kernel(samples_a, samples_b, gamma, coef)
    expect(kernel_mat.class).to eq(Numo::DFloat)
    expect(kernel_mat.shape[0]).to eq(n_samples_a)
    expect(kernel_mat.shape[1]).to eq(n_samples_b)
    expect(kernel_mat).to be_within(1.0e-8).of(kernel_mat_bf)
  end
end
