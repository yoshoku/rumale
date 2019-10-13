# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::StandardScaler do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand }

  it 'performs standardization of samples.' do
    normalizer = described_class.new
    normalized = normalizer.fit_transform(samples)
    mean_err = (normalized.mean(0) - Numo::DFloat.zeros(n_features)).abs.sum
    std_err = (normalized.stddev(0) - Numo::DFloat.ones(n_features)).abs.sum
    expect(mean_err).to be < 1.0e-8
    expect(std_err).to be < 1.0e-8
    expect(normalizer.mean_vec.class).to eq(Numo::DFloat)
    expect(normalizer.mean_vec.shape[0]).to eq(n_features)
    expect(normalizer.mean_vec.shape[1]).to be_nil
    expect(normalizer.std_vec.class).to eq(Numo::DFloat)
    expect(normalizer.std_vec.shape[0]).to eq(n_features)
    expect(normalizer.std_vec.shape[1]).to be_nil
  end

  it 'dumps and restores itself using Marshal module.' do
    transformer = described_class.new
    transformer.fit(samples)
    copied = Marshal.load(Marshal.dump(transformer))
    expect(transformer.mean_vec).to eq(copied.mean_vec)
    expect(transformer.std_vec).to eq(copied.std_vec)
  end
end
