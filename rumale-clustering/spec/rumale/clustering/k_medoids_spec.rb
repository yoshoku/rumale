# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Clustering::KMedoids do
  let(:three_clusters) { three_clusters_dataset }
  let(:x_mlt) { three_clusters[0] }
  let(:y_mlt) { three_clusters[1] }
  let(:dist_mat) { Rumale::PairwiseMetric.euclidean_distance(x_mlt) }
  let(:analyzer) { described_class.new(n_clusters: 3, max_iter: 100, random_seed: 1) }
  let(:analyzer_precomputed) { described_class.new(n_clusters: 3, metric: 'precomputed', max_iter: 100, random_seed: 1) }
  let(:non_learn_analyzer) { described_class.new(n_clusters: 3, init: 'k-means++', max_iter: 0, random_seed: 1) }

  it 'analyze cluster', :aggregate_failures do
    cluster_labels = analyzer.fit(x_mlt).predict(x_mlt)
    expect(cluster_labels).to be_a(Numo::Int32)
    expect(cluster_labels).to be_contiguous
    expect(cluster_labels.ndim).to eq(1)
    expect(cluster_labels.shape[0]).to eq(x_mlt.shape[0])
    expect(cluster_labels.eq(0).count).to eq(100)
    expect(cluster_labels.eq(1).count).to eq(100)
    expect(cluster_labels.eq(2).count).to eq(100)
    expect(analyzer.medoid_ids).to be_a(Numo::Int32)
    expect(analyzer.medoid_ids).to be_contiguous
    expect(analyzer.medoid_ids.ndim).to eq(1)
    expect(analyzer.medoid_ids.shape[0]).to eq(3)
    expect(analyzer.score(x_mlt, y_mlt)).to eq(1)
  end

  it 'analyze cluster with distance matrix', :aggregate_failures do
    analyzer_precomputed.fit(dist_mat)
    cluster_labels = analyzer_precomputed.predict(dist_mat[true, analyzer_precomputed.medoid_ids])
    expect(cluster_labels).to be_a(Numo::Int32)
    expect(cluster_labels).to be_contiguous
    expect(cluster_labels.ndim).to eq(1)
    expect(cluster_labels.shape[0]).to eq(x_mlt.shape[0])
    expect(cluster_labels.eq(0).count).to eq(100)
    expect(cluster_labels.eq(1).count).to eq(100)
    expect(cluster_labels.eq(2).count).to eq(100)
    expect(analyzer_precomputed.medoid_ids).to be_a(Numo::Int32)
    expect(analyzer_precomputed.medoid_ids).to be_contiguous
    expect(analyzer_precomputed.medoid_ids.ndim).to eq(1)
    expect(analyzer_precomputed.medoid_ids.shape[0]).to eq(3)
    expect(analyzer_precomputed.score(dist_mat, y_mlt)).to eq(1)
  end

  it 'raises ArgumentError when given a wrong shape matrix', :aggregate_failures do
    expect { analyzer_precomputed.fit(Numo::DFloat.new(4, 2).rand) }.to raise_error(ArgumentError)
    analyzer_precomputed.fit(dist_mat)
    expect { analyzer_precomputed.predict(Numo::DFloat.new(4, 10).rand) }.to raise_error(ArgumentError)
  end

  it 'initializes centroids with k-means++ algorithm' do
    expect(non_learn_analyzer.score(x_mlt, y_mlt)).to be >= 2.fdiv(3)
  end
end
