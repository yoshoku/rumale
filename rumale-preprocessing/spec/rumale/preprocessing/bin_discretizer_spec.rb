# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::BinDiscretizer do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand - 0.5 }
  let(:n_bins) { 8 }
  let(:discretizer) { described_class.new(n_bins: n_bins) }
  let(:transformed) { discretizer.fit_transform(samples) }

  it 'discretizes with k-bins', :aggregate_failures do
    expect(transformed.shape[0]).to eq(n_samples)
    expect(transformed.shape[1]).to eq(n_features)
    expect(transformed[true, 0].to_a.uniq.size).to be <= n_bins
    expect(discretizer.feature_steps).to be_a(Array)
    expect(discretizer.feature_steps[0]).to be_a(Numo::DFloat)
    expect(discretizer.feature_steps.size).to eq(n_features)
    expect(discretizer.feature_steps[0].size).to eq(n_bins)
  end
end
