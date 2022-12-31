# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::StratifiedShuffleSplit do
  let(:n_splits) { 3 }
  let(:n_samples) { 30 }
  let(:n_features) { 3 }
  let(:test_size) { 0.2 }
  let(:train_size) { 0.6 }
  let(:n_test_samples) { (n_samples * test_size).to_i }
  let(:n_train_samples) { (n_samples * train_size).to_i }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand }
  let(:labels) { Numo::Int32[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] }

  it 'splits the dataset with given test and train sizes', :aggregate_failures do
    splitter = described_class.new(n_splits: n_splits, test_size: test_size, train_size: train_size, random_seed: 1)
    validation_ids = splitter.split(samples, labels)
    expect(splitter.n_splits).to eq(n_splits)
    expect(validation_ids).to be_a(Array)
    expect(validation_ids.size).to eq(n_splits)
    expect(validation_ids[0].size).to eq(2)
    expect(validation_ids[0][0].size).to eq(n_train_samples)
    expect(validation_ids[0][1].size).to eq(n_test_samples)
    expect(validation_ids[1][0].size).to eq(n_train_samples)
    expect(validation_ids[1][1].size).to eq(n_test_samples)
    expect(validation_ids[2][0].size).to eq(n_train_samples)
    expect(validation_ids[2][1].size).to eq(n_test_samples)
    expect(labels[validation_ids[0][0]].eq(1).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[0][0]].eq(2).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[0][0]].eq(3).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[0][1]].eq(1).count).to eq((test_size * 10).to_i)
    expect(labels[validation_ids[0][1]].eq(2).count).to eq((test_size * 10).to_i)
    expect(labels[validation_ids[0][1]].eq(3).count).to eq((test_size * 10).to_i)
    expect(labels[validation_ids[1][0]].eq(1).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[1][0]].eq(2).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[1][0]].eq(3).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[1][1]].eq(1).count).to eq((test_size * 10).to_i)
    expect(labels[validation_ids[1][1]].eq(2).count).to eq((test_size * 10).to_i)
    expect(labels[validation_ids[1][1]].eq(3).count).to eq((test_size * 10).to_i)
    expect(labels[validation_ids[2][0]].eq(1).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[2][0]].eq(2).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[2][0]].eq(3).count).to eq((train_size * 10).to_i)
    expect(labels[validation_ids[2][1]].eq(1).count).to eq((test_size * 10).to_i)
    expect(labels[validation_ids[2][1]].eq(2).count).to eq((test_size * 10).to_i)
    expect(labels[validation_ids[2][1]].eq(3).count).to eq((test_size * 10).to_i)
  end

  it 'splits the dataset with given test size', :aggregate_failures do
    splitter = described_class.new(n_splits: n_splits, test_size: test_size, random_seed: 1)
    validation_ids = splitter.split(samples, labels)
    expect(splitter.n_splits).to eq(n_splits)
    expect(validation_ids).to be_a(Array)
    expect(validation_ids.size).to eq(n_splits)
    expect(validation_ids[0].size).to eq(2)
    expect(validation_ids[0][0].size).to eq(24)
    expect(validation_ids[0][1].size).to eq(6)
    expect(labels[validation_ids[0][0]].eq(1).count).to eq(8)
    expect(labels[validation_ids[0][0]].eq(2).count).to eq(8)
    expect(labels[validation_ids[0][0]].eq(3).count).to eq(8)
    expect(labels[validation_ids[0][1]].eq(1).count).to eq(2)
    expect(labels[validation_ids[0][1]].eq(2).count).to eq(2)
    expect(labels[validation_ids[0][1]].eq(3).count).to eq(2)
  end

  it 'raises ArgumentError given a wrong split number', :aggregate_failures do
    # exceeding the number of samples for each class
    splitter = described_class.new(n_splits: n_samples + 10)
    expect { splitter.split(samples, labels) }.to raise_error(ArgumentError)
    # less than 1
    splitter = described_class.new(n_splits: 0)
    expect { splitter.split(samples, labels) }.to raise_error(ArgumentError)
  end

  it 'raises RangeError given wrong sample sizes', :aggregate_failures do
    expect { described_class.new(n_splits: 1, test_size: 1.1).split(samples, labels) }.to raise_error(RangeError)
    expect { described_class.new(n_splits: 1, test_size: 0.0).split(samples, labels) }.to raise_error(RangeError)
    expect { described_class.new(n_splits: 1, train_size: 1.1).split(samples, labels) }.to raise_error(RangeError)
    expect { described_class.new(n_splits: 1, train_size: 0.01).split(samples, labels) }.to raise_error(RangeError)
    expect { described_class.new(n_splits: 1, train_size: 0.0).split(samples, labels) }.to raise_error(RangeError)
    expect do
      described_class.new(n_splits: 1, test_size: 0.1, train_size: 0.9).split(samples, labels)
    end.not_to raise_error
    expect do
      described_class.new(n_splits: 1, test_size: 0.2, train_size: 0.9).split(samples, labels)
    end.to raise_error(RangeError)
  end
end
