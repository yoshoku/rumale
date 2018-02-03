require 'spec_helper'

RSpec.describe SVMKit::ModelSelection::StratifiedKFold do
  let(:n_splits) { 3 }
  let(:n_samples) { 12 }
  let(:n_features) { 3 }
  let(:n_training_samples) { n_samples - n_samples / n_splits }
  let(:n_testing_samples) { n_samples / n_splits }
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand }
  let(:labels) { Numo::Int32[0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2] }

  it 'splits the dataset.' do
    splitter = described_class.new(n_splits: n_splits)
    validation_ids = splitter.split(samples, labels)
    expect(splitter.n_splits).to eq(n_splits)
    expect(validation_ids.class).to eq(Array)
    expect(validation_ids.size).to eq(n_splits)
    expect(validation_ids[0].size).to eq(2)
    expect(validation_ids[0][0].size).to eq(n_training_samples)
    expect(validation_ids[0][1].size).to eq(n_testing_samples)
    expect(validation_ids[0][0]).to match_array([2, 3, 4, 5, 7, 8, 10, 11])
    expect(validation_ids[0][1]).to match_array([0, 1, 6, 9])
    expect(validation_ids[1][0]).to match_array([0, 1, 4, 5, 6, 8, 9, 11])
    expect(validation_ids[1][1]).to match_array([2, 3, 7, 10])
    expect(validation_ids[2][0]).to match_array([0, 1, 2, 3, 6, 7, 9, 10])
    expect(validation_ids[2][1]).to match_array([4, 5, 8, 11])
  end

  it 'shuffles and splits the dataset.' do
    splitter = described_class.new(n_splits: n_splits, shuffle: true, random_seed: 1)
    validation_ids = splitter.split(samples, labels)
    expect(splitter.n_splits).to eq(n_splits)
    expect(validation_ids.class).to eq(Array)
    expect(validation_ids.size).to eq(n_splits)
    expect(validation_ids[0].size).to eq(2)
    expect(validation_ids[0][0].size).to eq(n_training_samples)
    expect(validation_ids[0][1].size).to eq(n_testing_samples)
    expect(validation_ids[0][0]).not_to match_array([2, 3, 4, 5, 7, 8, 10, 11])
    expect(validation_ids[0][1]).not_to match_array([0, 1, 6, 9])
  end

  it 'raises ArgumentError given a wrong split number.' do
    # exceeding the number of samples for each class
    splitter = described_class.new(n_splits: 4)
    expect { splitter.split(samples, labels) }.to raise_error(ArgumentError)
    # less than 2
    splitter = described_class.new(n_splits: 1)
    expect { splitter.split(samples, labels) }.to raise_error(ArgumentError)
  end
end
