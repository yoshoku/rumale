require 'spec_helper'

RSpec.describe SVMKit::Preprocessing::L2Normalizer do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) do
    rng = Random.new(1)
    rnd_vals = Array.new(n_samples * n_features) { rng.rand }
    NMatrix.new([n_samples, n_features], rnd_vals, dtype: :float64, stype: :dense)
  end

  it 'normalizes each sample to unit length.' do
    normalizer = described_class.new
    normalized = normalizer.fit_transform(samples)
    sum_norm = 0.0
    n_samples.times do |n|
      sum_norm += normalized.row(n).norm2
    end
    expect((sum_norm - n_samples).abs).to be < 1.0e-6
  end
end
