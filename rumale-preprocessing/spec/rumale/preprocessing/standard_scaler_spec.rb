# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::StandardScaler do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand }

  it 'performs standardization of samples', :aggregate_failures do
    normalizer = described_class.new
    normalized = normalizer.fit_transform(samples)
    mean_err = (normalized.mean(0) - Numo::DFloat.zeros(n_features)).abs.sum
    std_err = (normalized.stddev(0) - Numo::DFloat.ones(n_features)).abs.sum
    expect(mean_err).to be < 1.0e-8
    expect(std_err).to be < 1.0e-8
    expect(normalizer.mean_vec).to be_a(Numo::DFloat)
    expect(normalizer.mean_vec.ndim).to eq(1)
    expect(normalizer.mean_vec.shape[0]).to eq(n_features)
    expect(normalizer.std_vec).to be_a(Numo::DFloat)
    expect(normalizer.std_vec.ndim).to eq(1)
    expect(normalizer.std_vec.shape[0]).to eq(n_features)
  end
end
