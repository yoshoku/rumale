# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Clustering::SpectralClustering do
  let(:n_samples) { x.shape[0] }
  let(:n_cluster_samples) { n_samples.fdiv(n_clusters).to_i }
  let(:analyzer) { described_class.new(n_clusters: n_clusters, gamma: 32.0, random_seed: 1) }
  let(:cluster_labels) { analyzer.fit_predict(x) }

  describe 'three clusters dataset' do
    let(:three_clusters) { three_clusters_dataset }
    let(:x) { three_clusters[0] }
    let(:y) { three_clusters[1] }
    let(:n_clusters) { 3 }
    let(:copied) { Marshal.load(Marshal.dump(analyzer.fit(x))) }

    it 'analyzes clusters.', :aggregate_failures do
      expect(cluster_labels).to be_a(Numo::Int32)
      expect(cluster_labels).to be_contiguous
      expect(cluster_labels.ndim).to eq(1)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.eq(0).count).to eq(n_cluster_samples)
      expect(cluster_labels.eq(1).count).to eq(n_cluster_samples)
      expect(cluster_labels.eq(2).count).to eq(n_cluster_samples)
      expect(analyzer.embedding).to be_a(Numo::DFloat)
      expect(analyzer.embedding).to be_contiguous
      expect(analyzer.embedding.ndim).to eq(2)
      expect(analyzer.embedding.shape[0]).to eq(n_samples)
      expect(analyzer.embedding.shape[1]).to eq(n_clusters)
      expect(analyzer.labels).to be_a(Numo::Int32)
      expect(analyzer.labels).to be_contiguous
      expect(analyzer.labels.ndim).to eq(1)
      expect(analyzer.labels.shape[0]).to eq(n_samples)
      expect(analyzer.labels.eq(0).count).to eq(n_cluster_samples)
      expect(analyzer.labels.eq(1).count).to eq(n_cluster_samples)
      expect(analyzer.labels.eq(2).count).to eq(n_cluster_samples)
      expect(analyzer.score(x, y)).to eq(1)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(analyzer.class).to eq(copied.class)
      expect(analyzer.params).to eq(copied.params)
      expect(analyzer.embedding).to eq(copied.embedding)
      expect(analyzer.labels).to eq(copied.labels)
      expect(analyzer.score(x, y)).to eq(copied.score(x, y))
    end
  end

  describe 'two circles dataset' do
    let(:dataset) { Rumale::Dataset.make_circles(200, factor: 0.4, noise: 0.03, random_seed: 1) }
    let(:x) { dataset[0] }
    let(:y) { dataset[1] }
    let(:n_clusters) { 2 }

    it 'analyzes clusters.', :aggregate_failures do
      expect(cluster_labels).to be_a(Numo::Int32)
      expect(cluster_labels).to be_contiguous
      expect(cluster_labels.ndim).to eq(1)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.eq(0).count).to eq(n_cluster_samples)
      expect(cluster_labels.eq(1).count).to eq(n_cluster_samples)
      expect(analyzer.embedding).to be_a(Numo::DFloat)
      expect(analyzer.embedding).to be_contiguous
      expect(analyzer.embedding.ndim).to eq(2)
      expect(analyzer.embedding.shape[0]).to eq(n_samples)
      expect(analyzer.embedding.shape[1]).to eq(n_clusters)
      expect(analyzer.labels).to be_a(Numo::Int32)
      expect(analyzer.labels).to be_contiguous
      expect(analyzer.labels.ndim).to eq(1)
      expect(analyzer.labels.shape[0]).to eq(n_samples)
      expect(analyzer.labels.eq(0).count).to eq(n_cluster_samples)
      expect(analyzer.labels.eq(1).count).to eq(n_cluster_samples)
      expect(analyzer.score(x, y)).to eq(1)
    end
  end

  describe 'two moons dataset' do
    let(:dataset) { Rumale::Dataset.make_moons(200, noise: 0.03, random_seed: 1) }
    let(:x) { dataset[0] }
    let(:y) { dataset[1] }
    let(:n_clusters) { 2 }

    it 'analyzes clusters.', :aggregate_failures do
      expect(cluster_labels).to be_a(Numo::Int32)
      expect(cluster_labels).to be_contiguous
      expect(cluster_labels.ndim).to eq(1)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.eq(0).count).to eq(n_cluster_samples)
      expect(cluster_labels.eq(1).count).to eq(n_cluster_samples)
      expect(analyzer.embedding).to be_a(Numo::DFloat)
      expect(analyzer.embedding).to be_contiguous
      expect(analyzer.embedding.ndim).to eq(2)
      expect(analyzer.embedding.shape[0]).to eq(n_samples)
      expect(analyzer.embedding.shape[1]).to eq(n_clusters)
      expect(analyzer.labels).to be_a(Numo::Int32)
      expect(analyzer.labels).to be_contiguous
      expect(analyzer.labels.ndim).to eq(1)
      expect(analyzer.labels.shape[0]).to eq(n_samples)
      expect(analyzer.labels.eq(0).count).to eq(n_cluster_samples)
      expect(analyzer.labels.eq(1).count).to eq(n_cluster_samples)
      expect(analyzer.score(x, y)).to eq(1)
    end
  end
end
