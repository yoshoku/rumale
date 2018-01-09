require 'spec_helper'

RSpec.describe SVMKit::ModelSelection::KFold do
  let(:n_splits) { 3 }
  let(:n_samples) { 9 }
  let(:n_features) { 3 }
  let(:n_training_samples) { n_samples - n_samples / n_splits }
  let(:n_testing_samples) { n_samples / n_splits}
  let(:samples) { Numo::DFloat.new(n_samples, n_features).rand }

  it 'splits the dataset' do
    splitter = described_class.new(n_splits: n_splits)
    validation_ids = splitter.split(samples)
    expect(splitter.n_splits).to eq(n_splits)
    expect(validation_ids.class).to eq(Array)
    expect(validation_ids.size).to eq(n_splits)
    expect(validation_ids[0].size).to eq(2)
    expect(validation_ids[0][0].size).to eq(n_training_samples)
    expect(validation_ids[0][1].size).to eq(n_testing_samples)
    expect(validation_ids[0][0]).to match_array([3, 4, 5, 6, 7, 8])
    expect(validation_ids[0][1]).to match_array([0, 1, 2])
    expect(validation_ids[1][0]).to match_array([0, 1, 2, 6, 7, 8])
    expect(validation_ids[1][1]).to match_array([3, 4, 5])
    expect(validation_ids[2][0]).to match_array([0, 1, 2, 3, 4, 5])
    expect(validation_ids[2][1]).to match_array([6, 7, 8])
  end

  it 'shuffles and splits the dataset' do
    splitter = described_class.new(n_splits: n_splits, shuffle: true, random_seed: 1)
    validation_ids = splitter.split(samples)
    expect(splitter.n_splits).to eq(n_splits)
    expect(validation_ids.class).to eq(Array)
    expect(validation_ids.size).to eq(n_splits)
    expect(validation_ids[0].size).to eq(2)
    expect(validation_ids[0][0].size).to eq(n_training_samples)
    expect(validation_ids[0][1].size).to eq(n_testing_samples)
    expect(validation_ids[0][0]).not_to match_array([3, 4, 5, 6, 7, 8])
    expect(validation_ids[0][1]).not_to match_array([0, 1, 2])
  end

  it 'switches to the leave-one-out method given the wrong split number' do
    # exceeding the number of samples
    splitter = described_class.new(n_splits: n_samples + 10)
    validation_ids = splitter.split(samples)
    expect(splitter.n_splits).to eq(n_samples)
    expect(validation_ids.class).to eq(Array)
    expect(validation_ids.size).to eq(n_samples)
    expect(validation_ids[0][0].size).to eq(n_samples - 1)
    expect(validation_ids[0][1].size).to eq(1)
    expect(validation_ids[0][0]).to match_array([1, 2, 3, 4, 5, 6, 7, 8])
    expect(validation_ids[0][1]).to match_array([0])
    # less than 2
    splitter = described_class.new(n_splits: 1)
    validation_ids = splitter.split(samples)
    expect(splitter.n_splits).to eq(n_samples)
    expect(validation_ids.class).to eq(Array)
    expect(validation_ids.size).to eq(n_samples)
    expect(validation_ids[0][0].size).to eq(n_samples - 1)
    expect(validation_ids[0][1].size).to eq(1)
    expect(validation_ids[0][0]).to match_array([1, 2, 3, 4, 5, 6, 7, 8])
    expect(validation_ids[0][1]).to match_array([0])
  end
end
