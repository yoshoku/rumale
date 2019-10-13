# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Manifold::MDS do
  let(:x) { Marshal.load(File.read(__dir__ + '/../../test_samples.dat')) }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { 32 }
  let(:n_components) { 2 }
  let(:samples) { Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_components: n_features, random_seed: 1).fit_transform(x) }
  let(:distance_matrix) { Rumale::PairwiseMetric.euclidean_distance(samples) }
  let(:init_stress) { described_class.new(n_components: n_components, max_iter: 0, random_seed: 1).fit(samples).stress }
  let(:mds) { described_class.new(n_components: n_components, max_iter: 50, random_seed: 1) }
  let(:mds_metric) { described_class.new(n_components: n_components, metric: 'precomputed', max_iter: 50, random_seed: 1) }
  let(:mds_tol) { described_class.new(n_components: n_components, max_iter: 50, init: 'pca', tol: 100.0, random_seed: 1) }
  let(:mds_params) do
    described_class.new(n_components: n_components, max_iter: 100, init: 'pca', verbose: true, tol: nil, random_seed: 1)
  end

  it 'maps high-dimensional data into low-dimensional data.' do
    low_samples = mds.fit_transform(samples)
    expect(low_samples.class).to eq(Numo::DFloat)
    expect(low_samples.shape[0]).to eq(n_samples)
    expect(low_samples.shape[1]).to eq(n_components)
    expect(mds.embedding.class).to eq(Numo::DFloat)
    expect(mds.embedding.shape[0]).to eq(n_samples)
    expect(mds.embedding.shape[1]).to eq(n_components)
    expect(mds.n_iter).to eq(50)
    expect(mds.stress.class).to eq(Float)
    expect(mds.stress).not_to be_nil
    expect(mds.stress).to be < init_stress
  end

  it 'maps high-dimensional data represented by distance matrix.' do
    low_samples = mds_metric.fit_transform(distance_matrix)
    expect(low_samples.class).to eq(Numo::DFloat)
    expect(low_samples.shape[0]).to eq(n_samples)
    expect(low_samples.shape[1]).to eq(n_components)
    expect(mds_metric.embedding.class).to eq(Numo::DFloat)
    expect(mds_metric.embedding.shape[0]).to eq(n_samples)
    expect(mds_metric.embedding.shape[1]).to eq(n_components)
    expect(mds_metric.n_iter).to eq(50)
    expect(mds_metric.stress.class).to eq(Float)
    expect(mds_metric.stress).not_to be_nil
    expect(mds_metric.stress).to be < init_stress
  end

  it 'terminates optimization based on the tol parameter.' do
    low_samples = mds_tol.fit_transform(samples)
    expect(low_samples.class).to eq(Numo::DFloat)
    expect(low_samples.shape[0]).to eq(n_samples)
    expect(low_samples.shape[1]).to eq(n_components)
    expect(mds_tol.embedding.class).to eq(Numo::DFloat)
    expect(mds_tol.embedding.shape[0]).to eq(n_samples)
    expect(mds_tol.embedding.shape[1]).to eq(n_components)
    expect(mds_tol.n_iter).to be < 50
    expect(mds_tol.stress.class).to eq(Float)
    expect(mds_tol.stress).not_to be_nil
    expect(mds_tol.stress).to be < init_stress
  end

  # it 'raises ArgumentError when given a non-square matrix.' do
  #   expect { mds_metric.fit(Numo::DFloat.new(5, 3).rand) }.to raise_error(ArgumentError)
  # end

  it 'outputs debug messages.' do
    expect { mds_params.fit(samples) }.to output(/MDS/).to_stdout
  end

  it 'dumps and restores itself using Marshal module.' do
    copied = Marshal.load(Marshal.dump(mds_params))
    expect(mds_params.class).to eq(copied.class)
    expect(mds_params.params[:n_components]).to match(copied.params[:n_components])
    expect(mds_params.params[:max_iter]).to match(copied.params[:max_iter])
    expect(mds_params.params[:metric]).to match(copied.params[:metric])
    expect(mds_params.params[:init]).to match(copied.params[:init])
    expect(mds_params.params[:verbose]).to match(copied.params[:verbose])
    expect(mds_params.params[:random_seed]).to match(copied.params[:random_seed])
    expect(mds_params.embedding).to eq(copied.embedding)
    expect(mds_params.stress).to eq(copied.stress)
    expect(mds_params.n_iter).to eq(copied.n_iter)
    expect(mds_params.rng).to eq(copied.rng)
  end
end
