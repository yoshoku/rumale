# frozen_string_literal: true

require 'pp'
require 'spec_helper'

RSpec.describe Rumale::Clustering::DBSCAN do
  let(:x_mlt) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:y_mlt) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) - 1 }
  let(:analyzer) { described_class.new(eps: 1.0) }

  it 'analyze cluster.' do
    cluster_labels = analyzer.fit_predict(x_mlt)

    expect(cluster_labels.class).to eq(Numo::Int32)
    expect(cluster_labels.size).to eq(x_mlt.shape[0])
    expect(cluster_labels.shape[0]).to eq(x_mlt.shape[0])
    expect(cluster_labels.shape[1]).to be_nil
    expect(cluster_labels.eq(0).count).to eq(100)
    expect(cluster_labels.eq(1).count).to eq(100)
    expect(cluster_labels.eq(2).count).to eq(100)

    expect(analyzer.labels.class).to eq(Numo::Int32)
    expect(analyzer.labels.shape[0]).to eq(x_mlt.shape[0])
    expect(analyzer.labels.shape[1]).to be_nil

    expect(analyzer.core_sample_ids.class).to eq(Numo::Int32)
    expect(analyzer.core_sample_ids.shape[0]).not_to be_nil
    expect(analyzer.core_sample_ids.shape[1]).to be_nil

    expect(analyzer.score(x_mlt, y_mlt)).to eq(1)
  end

  it 'analyze outlier points.' do
    max_vec = x_mlt.max(0) * 2.0
    min_vec = x_mlt.min(0) * 2.0
    x = Numo::NArray.vstack([x_mlt, max_vec])
    x = Numo::NArray.vstack([x, min_vec])
    cluster_labels = analyzer.fit_predict(x)
    expect(cluster_labels.eq(-1).count).to eq(2)
    expect(analyzer.labels.eq(-1).count).to eq(2)
  end

  it 'dumps and restores itself using Marshal module.' do
    analyzer.fit(x_mlt)
    copied = Marshal.load(Marshal.dump(analyzer))
    expect(analyzer.class).to eq(copied.class)
    expect(analyzer.params[:eps]).to eq(copied.params[:eps])
    expect(analyzer.params[:min_samples]).to eq(copied.params[:min_samples])
    expect(analyzer.labels).to eq(copied.labels)
    expect(analyzer.core_sample_ids).to eq(copied.core_sample_ids)
    expect(analyzer.score(x_mlt, y_mlt)).to eq(copied.score(x_mlt, y_mlt))
  end
end
