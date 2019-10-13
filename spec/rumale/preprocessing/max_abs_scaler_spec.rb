# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::MaxAbsScaler do
  let(:samples) { Numo::DFloat[[0.8, 0.2], [-0.3, 0.5], [0.6, -0.9]] }
  let(:n_samples) { samples.shape[0] }
  let(:n_features) { samples.shape[1] }

  it 'normalizes range of features to [-1,1].' do
    normalizer = described_class.new
    normalized = normalizer.fit_transform(samples)
    expect(normalized.shape[0]).to eq(n_samples)
    expect(normalized.shape[1]).to eq(n_features)
    expect(normalized.min).to eq(-1)
    expect(normalized.max).to eq(1)
    expect(normalizer.max_abs_vec.class).to eq(Numo::DFloat)
    expect(normalizer.max_abs_vec.shape[0]).to eq(n_features)
    expect(normalizer.max_abs_vec.shape[1]).to be_nil
    expect(normalizer.max_abs_vec[0]).to eq(0.8)
    expect(normalizer.max_abs_vec[1]).to eq(0.9)
  end

  it 'dumps and restores itself using Marshal module.' do
    transformer = described_class.new
    transformer.fit(samples)
    copied = Marshal.load(Marshal.dump(transformer))
    expect(transformer.max_abs_vec).to eq(copied.max_abs_vec)
  end
end
