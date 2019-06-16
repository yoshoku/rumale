# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Clustering::PowerIteration do
  let(:n_samples) { x.shape[0] }
  let(:n_cluster_samples) { n_samples.fdiv(n_clusters).to_i }
  let(:analyzer) { described_class.new(n_clusters: n_clusters, gamma: 32.0, random_seed: 1) }
  let(:cluster_labels) { analyzer.fit_predict(x) }

  describe 'three clusters dataset' do
    let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
    let(:y) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) - 1 }
    let(:n_clusters) { 3 }
    let(:copied) { Marshal.load(Marshal.dump(analyzer.fit(x))) }

    it 'analyzes clusters.', aggregate_failures: true do
      expect(cluster_labels.class).to eq(Numo::Int32)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(n_cluster_samples)
      expect(cluster_labels.eq(1).count).to eq(n_cluster_samples)
      expect(cluster_labels.eq(2).count).to eq(n_cluster_samples)
      expect(analyzer.embedding.class).to eq(Numo::DFloat)
      expect(analyzer.embedding.shape[0]).to eq(n_samples)
      expect(analyzer.embedding.shape[1]).to be_nil
      expect(analyzer.n_iter).to be > 1
      expect(analyzer.score(x, y)).to eq(1)
    end

    it 'dumps and restores itself using Marshal module.', aggregate_failures: true do
      expect(analyzer.class).to eq(copied.class)
      expect(analyzer.params).to eq(copied.params)
      expect(analyzer.embedding).to eq(copied.embedding)
      expect(analyzer.n_iter).to eq(copied.n_iter)
      expect(analyzer.score(x, y)).to eq(copied.score(x, y))
    end
  end

  describe 'two circles dataset' do
    let(:dataset) { Rumale::Dataset.make_circles(200, factor: 0.4, noise: 0.03, random_seed: 1) }
    let(:x) { dataset[0] }
    let(:y) { dataset[1] }
    let(:n_clusters) { 2 }

    it 'analyzes clusters.', aggregate_failures: true do
      expect(cluster_labels.class).to eq(Numo::Int32)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(n_cluster_samples)
      expect(cluster_labels.eq(1).count).to eq(n_cluster_samples)
      expect(analyzer.embedding.class).to eq(Numo::DFloat)
      expect(analyzer.embedding.shape[0]).to eq(n_samples)
      expect(analyzer.embedding.shape[1]).to be_nil
      expect(analyzer.n_iter).to be > 1
      expect(analyzer.score(x, y)).to eq(1)
    end
  end

  describe 'two moons dataset' do
    let(:dataset) { Rumale::Dataset.make_moons(200, noise: 0.03, random_seed: 1) }
    let(:x) { dataset[0] }
    let(:y) { dataset[1] }
    let(:n_clusters) { 2 }

    it 'analyzes clusters.', aggregate_failures: true do
      expect(cluster_labels.class).to eq(Numo::Int32)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.shape[1]).to be_nil
      expect(cluster_labels.eq(0).count).to eq(n_cluster_samples)
      expect(cluster_labels.eq(1).count).to eq(n_cluster_samples)
      expect(analyzer.embedding.class).to eq(Numo::DFloat)
      expect(analyzer.embedding.shape[0]).to eq(n_samples)
      expect(analyzer.embedding.shape[1]).to be_nil
      expect(analyzer.n_iter).to be > 1
      expect(analyzer.score(x, y)).to eq(1)
    end
  end
end
