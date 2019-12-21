# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::VPTree do
  let(:dataset) { three_clusters_dataset }
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_neighbors) { 5 }
  let(:min_samples_leaf) { 1 }
  let(:vp_tree) { described_class.new(x, min_samples_leaf: min_samples_leaf) }
  let(:results) { vp_tree.query(x, n_neighbors) }
  let(:rel_ids) { results[0] }
  let(:rel_dists) { results[1] }
  let(:nn_labels) { y[rel_ids[true, 0]] }
  let(:copied) { Marshal.load(Marshal.dump(vp_tree)) }

  shared_examples 'k-nearest neighbor search' do
    it 'searches k-nearest neighbors', :aggregate_failures do
      expect(rel_ids.class).to eq(Numo::Int32)
      expect(rel_ids.ndim).to eq(2)
      expect(rel_ids.shape[0]).to eq(n_samples)
      expect(rel_ids.shape[1]).to eq(n_neighbors)
      expect(rel_dists.class).to eq(Numo::DFloat)
      expect(rel_dists.ndim).to eq(2)
      expect(rel_dists.shape[0]).to eq(n_samples)
      expect(rel_dists.shape[1]).to eq(n_neighbors)
      expect(nn_labels).to eq(y)
    end
  end

  context 'when parameter values are typical values' do
    it_behaves_like 'k-nearest neighbor search'

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(copied.params).to eq(vp_tree.params)
      expect(copied.data).to eq(vp_tree.data)
      expect(copied.query(x, n_neighbors)).to eq(results)
    end
  end

  context 'when n_neighbors parameter is large' do
    let(:n_neighbors) { 100 }

    it_behaves_like 'k-nearest neighbor search'
  end

  context 'when min_samples_leaf parameter is large' do
    let(:min_smaples_leaf) { 100 }

    it_behaves_like 'k-nearest neighbor search'
  end
end
