# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::BinDiscretizer do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand - 0.5 }
  let(:n_bins) { 8 }
  let(:discretizer) { described_class.new(n_bins: n_bins) }

  it 'discretizes with k-bins.' do
    transformed = discretizer.fit_transform(samples)
    expect(discretizer.feature_steps.class).to eq(Array)
    expect(discretizer.feature_steps[0].class).to eq(Numo::DFloat)
    expect(discretizer.feature_steps.size).to eq(n_features)
    expect(discretizer.feature_steps[0].size).to eq(n_bins)
    expect(transformed.shape[0]).to eq(n_samples)
    expect(transformed.shape[1]).to eq(n_features)
    expect(transformed[true, 0].to_a.uniq.size).to be <= n_bins
  end

  it 'dumps and restores itself using Marshal module.' do
    discretizer.fit(samples)
    copied = Marshal.load(Marshal.dump(discretizer))
    expect(discretizer.feature_steps).to eq(copied.feature_steps)
    expect(discretizer.params).to match(copied.params)
  end
end
