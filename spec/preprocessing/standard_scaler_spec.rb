require 'spec_helper'

RSpec.describe SVMKit::Preprocessing::StandardScaler do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) do
    rng = Random.new(1)
    rnd_vals = Array.new(n_samples * n_features) { rng.rand }
    NMatrix.new([n_samples, n_features], rnd_vals, dtype: :float64, stype: :dense)
  end

  it 'performs standardization of samples.' do
    normalizer = described_class.new
    normalized = normalizer.fit_transform(samples)
    mean_err = (normalized.mean(0) - NMatrix.zeros([1, n_features])).abs.sum(1)[0]
    std_err = (normalized.std(0) - NMatrix.ones([1, n_features])).abs.sum(1)[0]
    expect(mean_err).to be < 1.0e-8
    expect(std_err).to be < 1.0e-8
  end

  it 'dumps and restores itself using Marshal module.' do
    transformer = described_class.new
    transformer.fit(samples)
    copied = Marshal.load(Marshal.dump(transformer))
    expect(transformer.mean_vec).to eq(copied.mean_vec)
    expect(transformer.std_vec).to eq(copied.std_vec)
  end
end
