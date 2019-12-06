# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Clustering::GaussianMixture do
  let(:three_clusters) { three_clusters_dataset }
  let(:x) { three_clusters[0] }
  let(:y) { three_clusters[1] }
  let(:covariance_type) { 'diag' }
  let(:analyzer) { described_class.new(n_clusters: 3, covariance_type: covariance_type, max_iter: 50, tol: 0.0, random_seed: 1) }
  let(:cluster_labels) { analyzer.fit_predict(x) }
  let(:copied) { Marshal.load(Marshal.dump(analyzer.fit(x))) }

  context 'when covariance matrix is diagonal matrix' do
    it 'analyze cluster.', aggregate_failures: true do
      expect(cluster_labels.class).to eq(Numo::Int32)
      expect(cluster_labels.size).to eq(x.shape[0])
      expect(cluster_labels.shape[0]).to eq(x.shape[0])
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(100)
      expect(cluster_labels.eq(1).count).to eq(100)
      expect(cluster_labels.eq(2).count).to eq(100)
      expect(analyzer.weights.class).to eq(Numo::DFloat)
      expect(analyzer.weights.shape[0]).to eq(3)
      expect(analyzer.weights.shape[1]).to be_nil
      expect(analyzer.means.class).to eq(Numo::DFloat)
      expect(analyzer.means.shape[0]).to eq(3)
      expect(analyzer.means.shape[1]).to eq(2)
      expect(analyzer.covariances.class).to eq(Numo::DFloat)
      expect(analyzer.covariances.ndim).to eq(2)
      expect(analyzer.covariances.shape[0]).to eq(3)
      expect(analyzer.covariances.shape[1]).to eq(2)
      expect(analyzer.score(x, y)).to eq(1)
    end
  end

  context 'when covariance matrix is general covariance matrix' do
    let(:covariance_type) { 'full' }

    it 'analyze cluster.', aggregate_failures: true do
      expect(cluster_labels.class).to eq(Numo::Int32)
      expect(cluster_labels.size).to eq(x.shape[0])
      expect(cluster_labels.shape[0]).to eq(x.shape[0])
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(100)
      expect(cluster_labels.eq(1).count).to eq(100)
      expect(cluster_labels.eq(2).count).to eq(100)
      expect(analyzer.weights.class).to eq(Numo::DFloat)
      expect(analyzer.weights.shape[0]).to eq(3)
      expect(analyzer.weights.shape[1]).to be_nil
      expect(analyzer.means.class).to eq(Numo::DFloat)
      expect(analyzer.means.shape[0]).to eq(3)
      expect(analyzer.means.shape[1]).to eq(2)
      expect(analyzer.covariances.class).to eq(Numo::DFloat)
      expect(analyzer.covariances.ndim).to eq(3)
      expect(analyzer.covariances.shape[0]).to eq(3)
      expect(analyzer.covariances.shape[1]).to eq(2)
      expect(analyzer.covariances.shape[2]).to eq(2)
      expect(analyzer.score(x, y)).to eq(1)
    end

    context 'when Numo::Linalg is not loaded' do
      before do
        @backup = Numo::Linalg
        Numo.class_eval { remove_const(:Linalg) }
      end

      after { Numo::Linalg = @backup }

      it 'raises Runtime error' do
        expect { analyzer.fit(x) }.to raise_error(RuntimeError)
      end
    end
  end

  it 'dumps and restores itself using Marshal module.', aggregate_failures: true do
    expect(analyzer.class).to eq(copied.class)
    expect(analyzer.params[:n_clusters]).to eq(copied.params[:n_clusters])
    expect(analyzer.params[:init]).to eq(copied.params[:init])
    expect(analyzer.params[:covariance_type]).to eq(copied.params[:covariance_type])
    expect(analyzer.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(analyzer.params[:tol]).to eq(copied.params[:tol])
    expect(analyzer.params[:reg_covar]).to eq(copied.params[:reg_covar])
    expect(analyzer.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(analyzer.n_iter).to eq(copied.n_iter)
    expect(analyzer.weights).to eq(copied.weights)
    expect(analyzer.means).to eq(copied.means)
    expect(analyzer.covariances).to eq(copied.covariances)
    expect(analyzer.score(x, y)).to eq(copied.score(x, y))
  end
end
