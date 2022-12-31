# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::KernelMachine::KernelRidge do
  let(:x) { two_clusters_dataset[0] }
  let(:n_samples) { x.shape[0] }
  let(:kernel_mat) { Rumale::PairwiseMetric.rbf_kernel(x, nil, 1.0) }
  let(:reg_param) { 1.0 }
  let(:estimator) { described_class.new(reg_param: reg_param).fit(kernel_mat, y) }
  let(:predicted) { estimator.predict(kernel_mat) }
  let(:score) { estimator.score(kernel_mat, y) }

  context 'when single regression problem' do
    let(:y) { x[true, 0] + x[true, 1]**2 }

    it 'learns the model', :aggregate_failures do
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(1)
      expect(estimator.weight_vec.shape[0]).to eq(n_samples)
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end
  end

  context 'when multiple regression problem' do
    let(:y) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.8, 0.2]]) }
    let(:n_outputs) { y.shape[1] }

    it 'learns the model', :aggregate_failures do
      expect(estimator.weight_vec).to be_a(Numo::DFloat)
      expect(estimator.weight_vec).to be_contiguous
      expect(estimator.weight_vec.ndim).to eq(2)
      expect(estimator.weight_vec.shape[0]).to eq(n_samples)
      expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
      expect(predicted).to be_a(Numo::DFloat)
      expect(predicted).to be_contiguous
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.01).of(1.0)
    end

    context 'when given array to reg_param' do
      let(:reg_param) { Numo::DFloat[0.1, 0.5] }

      it 'learns the model', :aggregate_failures do
        expect(estimator.weight_vec).to be_a(Numo::DFloat)
        expect(estimator.weight_vec).to be_contiguous
        expect(estimator.weight_vec.ndim).to eq(2)
        expect(estimator.weight_vec.shape[0]).to eq(n_samples)
        expect(estimator.weight_vec.shape[1]).to eq(n_outputs)
        expect(estimator.params[:reg_param]).to eq(reg_param)
        expect(predicted).to be_a(Numo::DFloat)
        expect(predicted).to be_contiguous
        expect(predicted.ndim).to eq(2)
        expect(predicted.shape[0]).to eq(n_samples)
        expect(predicted.shape[1]).to eq(n_outputs)
        expect(score).to be_within(0.01).of(1.0)
      end
    end
  end
end
