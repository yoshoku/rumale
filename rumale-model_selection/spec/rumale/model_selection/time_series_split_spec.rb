# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::TimeSeriesSplit do
  let(:x) { Numo::DFloat.new(8, 2).rand }
  let(:y) { nil }
  let(:n_splits) { 3 }
  let(:splitter) { described_class.new(n_splits: n_splits, max_train_size: max_train_size) }
  let(:validation_ids) { splitter.split(x, y) }

  context 'when max_train_size is not given' do
    let(:max_train_size) { nil }

    it 'splits the dataset by time series style', :aggregate_failures do
      expect(splitter.n_splits).to eq(n_splits)
      expect(splitter.max_train_size).to be_nil
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids[0]).to contain_exactly([0, 1], [2, 3])
      expect(validation_ids[1]).to contain_exactly([0, 1, 2, 3], [4, 5])
      expect(validation_ids[2]).to contain_exactly([0, 1, 2, 3, 4, 5], [6, 7])
    end
  end

  context 'when max_train_size is given' do
    let(:max_train_size) { 1 }

    it 'splits the dataset by time series style with the specified training data size', :aggregate_failures do
      expect(splitter.n_splits).to eq(n_splits)
      expect(splitter.max_train_size).to eq(max_train_size)
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids[0][0].size).to eq(max_train_size)
      expect(validation_ids[1][0].size).to eq(max_train_size)
      expect(validation_ids[2][0].size).to eq(max_train_size)
    end
  end

  context 'when given a wrong split number' do
    it 'raises ArgumentError', :aggregate_failures do
      expect { described_class.new(n_splits: 0).split(x, y) }.to raise_error(ArgumentError)
      expect { described_class.new(n_splits: x.shape[0]).split(x, y) }.to raise_error(ArgumentError)
    end
  end
end
