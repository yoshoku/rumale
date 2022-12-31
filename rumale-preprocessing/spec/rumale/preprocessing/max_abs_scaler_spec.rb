# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::MaxAbsScaler do
  let(:samples) { Numo::DFloat[[0.8, 0.2], [-0.3, 0.5], [0.6, -0.9]] }
  let(:n_samples) { samples.shape[0] }
  let(:n_features) { samples.shape[1] }
  let(:normalizer) { described_class.new }
  let(:normalized) { normalizer.fit_transform(samples) }

  it 'normalizes range of features to [-1,1]', :aggregate_failures do
    expect(normalized.shape[0]).to eq(n_samples)
    expect(normalized.shape[1]).to eq(n_features)
    expect(normalized.min).to eq(-1)
    expect(normalized.max).to eq(1)
    expect(normalizer.max_abs_vec).to be_a(Numo::DFloat)
    expect(normalizer.max_abs_vec.ndim).to eq(1)
    expect(normalizer.max_abs_vec.shape[0]).to eq(n_features)
    expect(normalizer.max_abs_vec[0]).to eq(0.8)
    expect(normalizer.max_abs_vec[1]).to eq(0.9)
  end
end
