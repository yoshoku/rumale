# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::ModelSelection::GroupKFold do
  let(:x) { Numo::DFloat[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]] }
  let(:y) { nil }
  let(:groups) { Numo::Int32[0, 0, 1, 3, 1, 0, 3, 1, 0] }
  let(:splitter) { described_class.new(n_splits: n_splits) }
  let(:validation_ids) { splitter.split(x, y, groups) }

  context 'when the number of splits is equal to the number of groups' do
    let(:n_splits) { 3 }

    it 'splits the dataset with group labels', :aggregate_failures do
      expect(splitter.n_splits).to eq(n_splits)
      expect(validation_ids).to be_a(Array)
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids[0].size).to eq(2)
      expect(validation_ids[1].size).to eq(2)
      expect(validation_ids[2].size).to eq(2)
      expect(validation_ids[0][0]).to match_array(groups.ne(0).where.to_a)
      expect(validation_ids[0][1]).to match_array(groups.eq(0).where.to_a)
      expect(validation_ids[1][0]).to match_array(groups.ne(1).where.to_a)
      expect(validation_ids[1][1]).to match_array(groups.eq(1).where.to_a)
      expect(validation_ids[2][0]).to match_array(groups.ne(3).where.to_a)
      expect(validation_ids[2][1]).to match_array(groups.eq(3).where.to_a)
    end
  end

  context 'when given the number of splits is less than the number of groups' do
    let(:n_splits) { 2 }

    it 'splits the dataset with group labels', :aggregate_failures do
      expect(splitter.n_splits).to eq(n_splits)
      expect(validation_ids).to be_a(Array)
      expect(validation_ids.size).to eq(n_splits)
      expect(validation_ids[0].size).to eq(2)
      expect(validation_ids[1].size).to eq(2)
      expect(validation_ids[0][0]).to match_array(groups.ne(0).where.to_a)
      expect(validation_ids[0][1]).to match_array(groups.eq(0).where.to_a)
      expect(validation_ids[1][0]).to match_array(groups.eq(0).where.to_a)
      expect(validation_ids[1][1]).to match_array(groups.ne(0).where.to_a)
    end
  end

  context 'when given the number of splits is greater than the number of groups' do
    let(:n_splits) { 4 }

    it 'raises ArgumentError' do
      expect { splitter.split(x, y, groups) }.to raise_error(ArgumentError)
    end
  end
end
