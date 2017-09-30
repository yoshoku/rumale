require 'spec_helper'

RSpec.describe SVMKit::KernelApproximation::RBF do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) do
    rng = Random.new(1)
    rnd_vals = Array.new(n_samples * n_features) { rng.rand }
    NMatrix.new([n_samples, n_features], rnd_vals, dtype: :float64, stype: :dense)
  end

  it 'has a small approximation error for the RBF kernel function.' do
    # calculate RBF kernel matrix.
    kernel_matrix = NMatrix.zeros([n_samples, n_samples])
    n_samples.times do |m|
      n_samples.times do |n|
        distance = (samples.row(m) - samples.row(n)).norm2
        kernel_matrix[m, n] = Math.exp(-distance**2)
      end
    end
    # calculate approximate RBF kernel matrix.
    transformer = described_class.new(gamma: 1.0, n_components: 4096, random_seed: 1)
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
