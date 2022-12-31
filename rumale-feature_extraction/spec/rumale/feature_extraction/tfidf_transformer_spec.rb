# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::FeatureExtraction::TfidfTransformer do
  let(:x) { Numo::DFloat[[2, 0, 1], [0, 1, 3]] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:norm) { 'none' }
  let(:use_idf) { true }
  let(:smooth_idf) { false }
  let(:sublinear_tf) { false }
  let(:transformer) { described_class.new(norm: norm, use_idf: use_idf, smooth_idf: smooth_idf, sublinear_tf: sublinear_tf) }
  let(:z) { transformer.fit_transform(x) }

  context 'when using idf weights' do
    it 'returns the tf-idf representation', :aggregate_failures do
      expect(z).to be_a(Numo::DFloat)
      expect(z).to be_contiguous
      expect(z.ndim).to eq(2)
      expect(z.shape[0]).to eq(n_samples)
      expect(z.shape[1]).to eq(n_features)
      expect(transformer.idf).to be_a(Numo::DFloat)
      expect(transformer.idf).to be_contiguous
      expect(transformer.idf.ndim).to eq(1)
      expect(transformer.idf.shape[0]).to eq(n_features)
      expect((x * transformer.idf - z).abs.sum).to be < 1e-8
    end

    context 'with smoothed idf' do
      let(:smooth_idf) { true }
      let(:sidf) { Numo::NMath.log((n_samples + 1) / (Numo::DFloat[1, 1, 2] + 1)) + 1 }

      it 'returns the tf-idf representation with smoothed idf' do
        expect((sidf - transformer.fit(x).idf).abs.sum).to be < 1e-8
      end
    end
  end

  context 'when not using idf weights' do
    let(:use_idf) { false }

    it 'returns the vectors withou idf weighting', :aggregate_failures do
      expect(transformer.idf).to be_nil
      expect((x - z).abs.sum).to be < 1e-8
    end

    context 'with sublinear tf' do
      let(:sublinear_tf) { true }
      let(:inv_z) do
        z.dup.tap { |inv_z| inv_z[inv_z.ne(0)] = Numo::NMath.exp(inv_z[inv_z.ne(0)] - 1) }
      end

      it 'returns the vectors applied sublinear tf scaling' do
        expect((x - inv_z).abs.sum).to be < 1e-8
      end
    end
  end

  context 'when normalizing by l2 norm' do
    let(:norm) { 'l2' }

    it 'normalizes transformed vectors by l2 norm' do
      expect(z.dot(z.transpose).diagonal.sum).to eq(n_samples)
    end
  end

  context 'when normalizing by l1 norm' do
    let(:norm) { 'l1' }

    it 'normalizes transformed vectors by l1 norm' do
      expect(z.abs.sum).to eq(n_samples)
    end
  end

  context 'when not normalizing' do
    let(:norm) { 'none' }

    it 'does not normalize transformed vectors', :aggregate_failures do
      expect(z.dot(z.transpose).diagonal.sum).not_to eq(n_samples)
      expect(z.abs.sum).not_to eq(n_samples)
    end
  end
end
