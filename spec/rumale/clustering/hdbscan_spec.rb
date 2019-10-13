# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Clustering::HDBSCAN do
  let(:dataset) { Rumale::Dataset.make_moons(200, noise: 0.03, random_seed: 1) }
  let(:samples) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { samples.shape[0] }
  let(:metric) { 'euclidean' }
  let(:analyzer) { described_class.new(min_samples: 5, metric: metric) }
  let(:cluster_labels) { analyzer.fit_predict(x) }
  let(:n_outliers) { 10 }
  let(:samples_with_outlier) do
    outliers = (Rumale::Utils.rand_uniform([n_outliers, 2], Random.new(10)) - 0.5) * 5.0
    Numo::NArray.vstack([samples, outliers])
  end

  shared_examples 'cluster analysis' do
    it 'finds clusters.', aggregate_failures: true do
      expect(cluster_labels.class).to eq(Numo::Int32)
      expect(cluster_labels.size).to eq(n_samples)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(100)
      expect(cluster_labels.eq(1).count).to eq(100)
      expect(analyzer.labels.class).to eq(Numo::Int32)
      expect(analyzer.labels.shape[0]).to eq(n_samples)
      expect(analyzer.labels.shape[1]).to be_nil
      expect(analyzer.score(x, y)).to eq(1)
    end
  end

  shared_examples 'outlier detection' do
    it 'detects outlier points.', aggregate_failures: true do
      expect(cluster_labels.eq(-1).count).to eq(n_outliers)
      expect(analyzer.labels.eq(-1).count).to eq(n_outliers)
    end
  end

  context "when metric is 'euclidean'" do
    let(:metric) { 'euclidean' }

    context 'when dataset does not contain outlier' do
      let(:x) { samples }

      it_behaves_like 'cluster analysis'
    end

    context 'when dataset contains outlier' do
      let(:x) { samples_with_outlier }

      it_behaves_like 'outlier detection'
    end
  end

  context "when metric is 'precomputed'" do
    let(:metric) { 'precomputed' }

    context 'when dataset does not contain outlier' do
      let(:x) { Rumale::PairwiseMetric.manhattan_distance(samples) }

      it_behaves_like 'cluster analysis'
    end

    context 'when dataset contains outlier' do
      let(:x) { Rumale::PairwiseMetric.manhattan_distance(samples_with_outlier) }

      it_behaves_like 'outlier detection'
    end

    it 'raises ArgumentError when given a non-square matrix', aggregate_failures: true do
      expect { analyzer.fit(Numo::DFloat.new(3, 2).rand) }.to raise_error(ArgumentError)
      expect { analyzer.fit_predict(Numo::DFloat.new(2, 3).rand) }.to raise_error(ArgumentError)
    end
  end

  it 'dumps and restores itself using Marshal module.', aggregate_failures: true do
    analyzer.fit(samples)
    copied = Marshal.load(Marshal.dump(analyzer))
    expect(analyzer.class).to eq(copied.class)
    expect(analyzer.params[:min_samples]).to eq(copied.params[:min_samples])
    expect(analyzer.params[:min_cluster_size]).to eq(copied.params[:min_cluster_size])
    expect(analyzer.params[:metric]).to eq(copied.params[:metric])
    expect(analyzer.labels).to eq(copied.labels)
    expect(analyzer.score(samples, y)).to eq(copied.score(samples, y))
  end
end
