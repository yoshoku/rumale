# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Utils do
  let(:rng) { Random.new(42) }

  describe '#choice_ids' do
    let(:ids) { described_class.choice_ids(5, [0.2, 0.2, 0.6, 0, 0], rng) }

    it 'choices index randomly according to given probabilities', :aggregate_failures do
      expect(ids).to be_a(Array)
      expect(ids.count(0)).to eq(1)
      expect(ids.count(1)).to eq(1)
      expect(ids.count(2)).to eq(3)
      expect(ids.count(3)).to be_zero
      expect(ids.count(4)).to be_zero
    end
  end

  describe '#rand_uniform' do
    let(:rand_vec) { described_class.rand_uniform(1000, rng) }
    let(:rand_mat) { described_class.rand_uniform([50, 20], rng) }

    it 'generates uniform random array with given shape', :aggregate_failures do
      expect(rand_vec).to be_a(Numo::DFloat)
      expect(rand_mat).to be_a(Numo::DFloat)
      expect(rand_vec.ndim).to eq(1)
      expect(rand_mat.ndim).to eq(2)
      expect(rand_vec.shape[0]).to eq(1000)
      expect(rand_mat.shape[0]).to eq(50)
      expect(rand_mat.shape[1]).to eq(20)
      expect(rand_vec.mean).to be_within(0.1).of(0.5)
      expect(rand_mat.mean).to be_within(0.1).of(0.5)
    end
  end

  describe '#rand_normal' do
    let(:rand_vec) { described_class.rand_normal(1000, rng) }
    let(:rand_mat) { described_class.rand_normal([50, 20], rng) }

    it 'generates gaussian random array with given shape', :aggregate_failures do
      expect(rand_vec).to be_a(Numo::DFloat)
      expect(rand_mat).to be_a(Numo::DFloat)
      expect(rand_vec.ndim).to eq(1)
      expect(rand_mat.ndim).to eq(2)
      expect(rand_vec.shape[0]).to eq(1000)
      expect(rand_mat.shape[0]).to eq(50)
      expect(rand_mat.shape[1]).to eq(20)
      expect(rand_vec.mean).to be_within(0.1).of(0.0)
      expect(rand_mat.mean).to be_within(0.1).of(0.0)
      expect(rand_vec.stddev).to be_within(0.1).of(1.0)
      expect(rand_mat.stddev).to be_within(0.1).of(1.0)
    end
  end

  describe '#binarize_labels' do
    let(:binarized) { described_class.binarize_labels(Numo::Int32[0, -1, 3, 3, 1, 1]) }

    it 'generates binary matrix according to given labels', :aggregate_failures do
      expect(binarized).to be_a(Numo::Int32)
      expect(binarized.ndim).to eq(2)
      expect(binarized.shape[0]).to eq(6)
      expect(binarized.shape[1]).to eq(4)
      expect(binarized).to eq(Numo::Int32[
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0]
      ])
    end
  end

  describe '#normalize' do
    let(:n_samples) { 10 }
    let(:x) { Numo::DFloat.new(n_samples, 5).rand - 0.5 }
    let(:normalized) { described_class.normalize(x, norm) }

    context 'when using L2-norm' do
      let(:norm) { 'l2' }

      it 'normalizes each data point with L2-norm', :aggregate_failures do
        expect(normalized).to be_a(Numo::DFloat)
        expect(normalized).to be_contiguous
        expect(normalized.ndim).to eq(2)
        expect(normalized.shape[0]).to eq(n_samples)
        expect(normalized.shape[1]).to eq(5)
        expect((normalized.dot(normalized.transpose).trace - n_samples).abs).to be < 1.0e-6
      end
    end

    context 'when using L1-norm' do
      let(:norm) { 'l1' }

      it 'normalizes each data point with L1-norm', :aggregate_failures do
        expect(normalized).to be_a(Numo::DFloat)
        expect(normalized).to be_contiguous
        expect(normalized.ndim).to eq(2)
        expect(normalized.shape[0]).to eq(n_samples)
        expect(normalized.shape[1]).to eq(5)
        expect((normalized.abs.sum - n_samples).abs).to be < 1.0e-6
      end
    end

    context 'when given an unsupported norm type' do
      it 'raises ArgumentError' do
        expect { described_class.normalize(x, 'max') }.to raise_error(ArgumentError)
      end
    end
  end
end
