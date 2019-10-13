# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::MinMaxScaler do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand }
  let(:sparse_samples) do
    x = Numo::DFloat.new(n_samples, 128).rand - 0.8
    0.5 * (x + x.abs)
  end

  it 'normalizes range of features to [0,1].' do
    normalizer = described_class.new
    normalized = normalizer.fit_transform(samples)
    expect(normalized.min).to eq(0)
    expect(normalized.max).to eq(1)
    expect(normalizer.min_vec.class).to eq(Numo::DFloat)
    expect(normalizer.min_vec.shape[0]).to eq(n_features)
    expect(normalizer.min_vec.shape[1]).to be_nil
    expect(normalizer.max_vec.class).to eq(Numo::DFloat)
    expect(normalizer.max_vec.shape[0]).to eq(n_features)
    expect(normalizer.max_vec.shape[1]).to be_nil
  end

  it 'normalizes range of features to a given range.' do
    normalizer = described_class.new(feature_range: [-3, 2])
    normalized = normalizer.fit_transform(samples)
    expect(normalized.min).to eq(-3)
    expect(normalized.max).to eq(2)
  end

  it 'normalizes sparse samples.' do
    normalizer = described_class.new(feature_range: [-1, 1])
    normalized = normalizer.fit_transform(sparse_samples)
    expect(normalized.isnan.count).to be_zero
    expect(normalized.min).to eq(-1)
    expect(normalized.max).to eq(1)
  end

  it 'dumps and restores itself using Marshal module.' do
    transformer = described_class.new
    transformer.fit(samples)
    copied = Marshal.load(Marshal.dump(transformer))
    expect(transformer.min_vec).to eq(copied.min_vec)
    expect(transformer.max_vec).to eq(copied.max_vec)
    expect(transformer.params[:feature_range][0]).to eq(copied.params[:feature_range][0])
    expect(transformer.params[:feature_range][1]).to eq(copied.params[:feature_range][1])
  end
end
