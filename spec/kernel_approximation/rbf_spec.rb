require 'spec_helper'

RSpec.describe SVMKit::KernelApproximation::RBF do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:n_components) { 4096 }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand }
  let(:kernel_matrix) do
    kernel_matrix = Numo::DFloat.zeros(n_samples, n_samples)
    n_samples.times do |m|
      n_samples.times do |n|
        distance = Math.sqrt(((samples[m, true] - samples[n, true])**2).sum)
        kernel_matrix[m, n] = Math.exp(-distance**2)
      end
    end
    kernel_matrix
  end

  it 'has a small approximation error for the RBF kernel function.' do
    # calculate approximate RBF kernel matrix.
    transformer = described_class.new(gamma: 1.0, n_components: n_components, random_seed: 1)
    new_samples = transformer.fit_transform(samples)
    inner_matrix = new_samples.dot(new_samples.transpose)
    # evalute mean error.
    mean_error = 0.0
    n_samples.times do |m|
      n_samples.times do |n|
        mean_error += ((kernel_matrix[m, n] - inner_matrix[m, n])**2)**0.5
      end
    end
    mean_error /= n_samples * n_samples
    expect(mean_error).to be < 0.01
    expect(transformer.random_mat.class).to eq(Numo::DFloat)
    expect(transformer.random_mat.shape[0]).to eq(n_features)
    expect(transformer.random_mat.shape[1]).to eq(n_components)
    expect(transformer.random_vec.class).to eq(Numo::DFloat)
    expect(transformer.random_vec.shape[0]).to eq(n_components)
    expect(transformer.random_vec.shape[1]).to be_nil
  end

  it 'dumps and restores itself using Marshal module.' do
    transformer = described_class.new(gamma: 1.0, n_components: 128, random_seed: 1)
    transformer.fit(samples)
    copied = Marshal.load(Marshal.dump(transformer))
    expect(transformer.class).to eq(copied.class)
    expect(transformer.params[:gamma]).to eq(copied.params[:gamma])
    expect(transformer.params[:n_components]).to eq(copied.params[:n_components])
    expect(transformer.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(transformer.random_mat).to eq(copied.random_mat)
    expect(transformer.random_vec).to eq(copied.random_vec)
    expect(transformer.rng).to eq(copied.rng)
  end
end
