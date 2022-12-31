# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Clustering::GaussianMixture do
  let(:three_clusters) { three_clusters_dataset }
  let(:x) { three_clusters[0] }
  let(:y) { three_clusters[1] }
  let(:covariance_type) { 'diag' }
  let(:analyzer) do
    described_class.new(n_clusters: 3, covariance_type: covariance_type, max_iter: 50, tol: 0.0, random_seed: 1)
  end
  let(:cluster_labels) { analyzer.fit_predict(x) }
  let(:copied) { Marshal.load(Marshal.dump(analyzer.fit(x))) }

  context 'when covariance matrix is diagonal matrix' do
    it 'analyze cluster', :aggregate_failures do
      expect(cluster_labels).to be_a(Numo::Int32)
      expect(cluster_labels).to be_contiguous
      expect(cluster_labels.size).to eq(x.shape[0])
      expect(cluster_labels.shape[0]).to eq(x.shape[0])
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(100)
      expect(cluster_labels.eq(1).count).to eq(100)
      expect(cluster_labels.eq(2).count).to eq(100)
      expect(analyzer.weights).to be_a(Numo::DFloat)
      expect(analyzer.weights).to be_contiguous
      expect(analyzer.weights.shape[0]).to eq(3)
      expect(analyzer.weights.shape[1]).to be_nil
      expect(analyzer.means).to be_a(Numo::DFloat)
      expect(analyzer.means).to be_contiguous
      expect(analyzer.means.shape[0]).to eq(3)
      expect(analyzer.means.shape[1]).to eq(2)
      expect(analyzer.covariances).to be_a(Numo::DFloat)
      expect(analyzer.covariances).to be_contiguous
      expect(analyzer.covariances.ndim).to eq(2)
      expect(analyzer.covariances.shape[0]).to eq(3)
      expect(analyzer.covariances.shape[1]).to eq(2)
      expect(analyzer.score(x, y)).to eq(1)
    end
  end

  context 'when covariance matrix is general covariance matrix' do
    let(:covariance_type) { 'full' }

    it 'analyze cluster', :aggregate_failures do
      expect(cluster_labels).to be_a(Numo::Int32)
      expect(cluster_labels).to be_contiguous
      expect(cluster_labels.size).to eq(x.shape[0])
      expect(cluster_labels.shape[0]).to eq(x.shape[0])
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(100)
      expect(cluster_labels.eq(1).count).to eq(100)
      expect(cluster_labels.eq(2).count).to eq(100)
      expect(analyzer.weights).to be_a(Numo::DFloat)
      expect(analyzer.weights).to be_contiguous
      expect(analyzer.weights.shape[0]).to eq(3)
      expect(analyzer.weights.shape[1]).to be_nil
      expect(analyzer.means).to be_a(Numo::DFloat)
      expect(analyzer.means).to be_contiguous
      expect(analyzer.means.shape[0]).to eq(3)
      expect(analyzer.means.shape[1]).to eq(2)
      expect(analyzer.covariances).to be_a(Numo::DFloat)
      expect(analyzer.covariances).to be_contiguous
      expect(analyzer.covariances.ndim).to eq(3)
      expect(analyzer.covariances.shape[0]).to eq(3)
      expect(analyzer.covariances.shape[1]).to eq(2)
      expect(analyzer.covariances.shape[2]).to eq(2)
      expect(analyzer.score(x, y)).to eq(1)
    end

    context 'when Numo::Linalg is not loaded' do
      before { hide_const('Numo::Linalg') }

      it 'raises Runtime error' do
        expect { analyzer.fit(x) }.to raise_error(RuntimeError)
      end
    end
  end
end
