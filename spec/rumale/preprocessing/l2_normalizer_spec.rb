# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::L2Normalizer do
  let(:n_samples) { 10 }
  let(:n_features) { 4 }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand }

  it 'normalizes each sample to unit length.' do
    normalizer = described_class.new
    normalized = normalizer.fit_transform(samples)
    dot_mat = normalized.dot(normalized.transpose)
    sum_norm = Array.new(n_samples) { |n| dot_mat[n, n] }.inject(:+)
    expect((sum_norm - n_samples).abs).to be < 1.0e-6
    expect(normalizer.norm_vec.class).to eq(Numo::DFloat)
    expect(normalizer.norm_vec.shape[0]).to eq(n_samples)
    expect(normalizer.norm_vec.shape[1]).to be_nil
  end
end
