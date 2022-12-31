# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::GroupShuffleSplit do
  let(:x) { Numo::DFloat.new(9, 2).rand }
  let(:y) { nil }
  let(:groups) { Numo::Int32[0, 0, 1, 3, 1, 0, 3, 1, 3] }
  let(:classes) { groups.to_a.uniq.sort }
  let(:n_splits) { 2 }
  let(:splitter) do
    described_class.new(n_splits: n_splits, test_size: test_size, train_size: train_size, random_seed: 1)
  end
  let(:validation_ids) { splitter.split(x, y, groups) }
  let(:train_groups0) { groups[validation_ids[0][0]].to_a.uniq.sort }
  let(:test_groups0) { groups[validation_ids[0][1]].to_a.uniq.sort }
  let(:train_groups1) { groups[validation_ids[1][0]].to_a.uniq.sort }
  let(:test_groups1) { groups[validation_ids[1][1]].to_a.uniq.sort }

  context 'when given appropriate test_size' do
    let(:test_size) { 0.2 }
    let(:train_size) { nil }

    it 'splits the dataset with group labels', :aggregate_failures do
      expect(splitter.n_splits).to eq(n_splits)
      expect(validation_ids).to be_a(Array)
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids[0].size).to eq(2)
      expect(validation_ids[1].size).to eq(2)
      expect(train_groups0.size).to eq(2)
      expect(test_groups0.size).to eq(1)
      expect(classes - train_groups0).to match(test_groups0)
      expect(classes - test_groups0).to match(train_groups0)
      expect(train_groups1.size).to eq(2)
      expect(test_groups1.size).to eq(1)
      expect(classes - train_groups1).to match(test_groups1)
      expect(classes - test_groups1).to match(train_groups1)
    end
  end

  context 'when given appropriate train_size' do
    let(:test_size) { 0.2 }
    let(:train_size) { 0.6 }

    it 'splits the dataset with group labels', :aggregate_failures do
      expect(train_groups0.size).to eq(1)
      expect(test_groups0.size).to eq(1)
      expect(train_groups1.size).to eq(1)
      expect(test_groups1.size).to eq(1)
    end
  end

  context 'when given wrong test and train _size' do
    it 'raises RangeError', :aggregate_failures do
      expect { described_class.new(test_size: 0).split(x, y, groups) }.to raise_error(RangeError)
      expect { described_class.new(test_size: 1.1).split(x, y, groups) }.to raise_error(RangeError)
      expect { described_class.new(train_size: 0).split(x, y, groups) }.to raise_error(RangeError)
      expect { described_class.new(train_size: 1.1).split(x, y, groups) }.to raise_error(RangeError)
      expect { described_class.new(test_size: 0.5, train_size: 1).split(x, y, groups) }.to raise_error(RangeError)
      expect { described_class.new(test_size: 0.2, train_size: 0.8).split(x, y, groups) }.not_to raise_error
    end
  end
end
