# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Clustering::SNN do
  let(:dataset) { Rumale::Dataset.make_circles(200, factor: 0.4, noise: 0.03, random_seed: 1) }
  let(:samples) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { samples.shape[0] }
  let(:metric) { 'euclidean' }
  let(:analyzer) { described_class.new(n_neighbors: 18, eps: 10, min_samples: 10, metric: metric) }
  let(:cluster_labels) { analyzer.fit_predict(x) }
  let(:n_outliers) { 5 }
  let(:samples_with_outlier) do
    outliers = Rumale::Utils.rand_normal([n_outliers, 2], Random.new(1), 0, 0.01)
    Numo::NArray.vstack([samples, outliers])
  end

  shared_examples 'cluster analysis' do
    it 'finds clusters.', :aggregate_failures do
      expect(cluster_labels).to be_a(Numo::Int32)
      expect(cluster_labels).to be_contiguous
      expect(cluster_labels.ndim).to eq(1)
      expect(cluster_labels.shape[0]).to eq(n_samples)
      expect(cluster_labels.eq(0).count).to eq(100)
      expect(cluster_labels.eq(1).count).to eq(100)
      expect(analyzer.labels).to be_a(Numo::Int32)
      expect(analyzer.labels).to be_contiguous
      expect(analyzer.labels.ndim).to eq(1)
      expect(analyzer.labels.shape[0]).to eq(n_samples)
      expect(analyzer.core_sample_ids).to be_a(Numo::Int32)
      expect(analyzer.core_sample_ids).to be_contiguous
      expect(analyzer.core_sample_ids.ndim).to eq(1)
      expect(analyzer.score(x, y)).to eq(1)
    end
  end

  shared_examples 'outlier detection' do
    it 'detects outlier points.' do
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

    it 'raises ArgumentError when given a non-square matrix' do
      expect { analyzer.fit(Numo::DFloat.new(3, 2).rand) }.to raise_error(ArgumentError)
      expect { analyzer.fit_predict(Numo::DFloat.new(2, 3).rand) }.to raise_error(ArgumentError)
    end
  end

  it 'dumps and restores itself using Marshal module.' do
    analyzer.fit(samples)
    copied = Marshal.load(Marshal.dump(analyzer))
    expect(analyzer.class).to eq(copied.class)
    expect(analyzer.params[:n_neighbors]).to eq(copied.params[:n_neighbors])
    expect(analyzer.params[:eps]).to eq(copied.params[:eps])
    expect(analyzer.params[:min_samples]).to eq(copied.params[:min_samples])
    expect(analyzer.params[:metric]).to eq(copied.params[:metric])
    expect(analyzer.labels).to eq(copied.labels)
    expect(analyzer.core_sample_ids).to eq(copied.core_sample_ids)
    expect(analyzer.score(samples, y)).to eq(copied.score(samples, y))
  end
end
