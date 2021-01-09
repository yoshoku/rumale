# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection do
  describe '#train_test_split' do
    let(:n_samples) { 30 }
    let(:test_size) { 0.2 }
    let(:x) { Numo::DFloat.new(n_samples, 2).rand }
    let(:y) { Numo::Int32[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] }
    let(:res) { described_class.train_test_split(x, y, test_size: test_size, stratify: stratify, random_seed: 1) }
    let(:n_test_samples) { (n_samples * test_size).to_i }
    let(:n_train_samples) { (n_samples * (1.0 - test_size)).to_i }
    let(:x_train) { res[0] }
    let(:x_test) { res[1] }
    let(:y_train) { res[2] }
    let(:y_test) { res[3] }

    context 'when stratify is false' do
      let(:stratify) { false }

      it 'splits the dataset given test size.', :aggregate_failures do
        expect(res).to be_a(Array)
        expect(res.size).to eq(4)
        expect(x_train.shape[0]).to eq(n_train_samples)
        expect(x_test.shape[0]).to eq(n_test_samples)
        expect(y_train.shape[0]).to eq(n_train_samples)
        expect(y_test.shape[0]).to eq(n_test_samples)
      end
    end

    context 'when stratify is true' do
      let(:stratify) { true }

      it 'splits the dataset given test size in a stratified fashion.', :aggregate_failures do
        expect(res).to be_a(Array)
        expect(res.size).to eq(4)
        expect(x_train.shape[0]).to eq(n_train_samples)
        expect(x_test.shape[0]).to eq(n_test_samples)
        expect(y_train.shape[0]).to eq(n_train_samples)
        expect(y_test.shape[0]).to eq(n_test_samples)
        expect(y_train.eq(1).count).to eq(8)
        expect(y_train.eq(2).count).to eq(8)
        expect(y_train.eq(3).count).to eq(8)
        expect(y_test.eq(1).count).to eq(2)
        expect(y_test.eq(2).count).to eq(2)
        expect(y_test.eq(3).count).to eq(2)
      end
    end
  end
end
