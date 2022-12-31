# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Validation do
  let(:arr1d) { [1, 2, 3, 4] }
  let(:arr2d) { [[1, 2], [3, 4]] }
  let(:arr3d) { [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] }

  describe '#check_convert_sample_array' do
    it 'converts given invalid type array into dfloat array' do
      expect(described_class.check_convert_sample_array([[1, 2, 3], [4, 5, 6]])).to eq(Numo::DFloat[[1, 2, 3], [4, 5, 6]])
    end

    it 'raises ArgumentError when given invalid shape array', :aggregate_failures do
      expect { described_class.check_convert_sample_array(Numo::DFloat.cast(arr1d)) }.to raise_error(ArgumentError)
      expect { described_class.check_convert_sample_array(Numo::DFloat.cast(arr2d)) }.not_to raise_error
      expect { described_class.check_convert_sample_array(Numo::DFloat.cast(arr3d)) }.to raise_error(ArgumentError)
    end
  end

  describe '#check_convert_label_array' do
    it 'converts given invalid type array into integer array' do
      expect(described_class.check_convert_label_array([1, 2, 3])).to eq(Numo::Int32[1, 2, 3])
    end

    it 'raises ArgumentError when given invalid shape array', :aggregate_failures do
      expect { described_class.check_convert_label_array(Numo::Int32.cast(arr1d)) }.not_to raise_error
      expect { described_class.check_convert_label_array(Numo::Int32.cast(arr2d)) }.to raise_error(ArgumentError)
      expect { described_class.check_convert_label_array(Numo::Int32.cast(arr3d)) }.to raise_error(ArgumentError)
    end
  end

  describe '#check_convert_target_value_array' do
    it 'converts given invalid type array into dfloat array' do
      expect(described_class.check_convert_target_value_array([1, 2, 3])).to eq(Numo::DFloat[1, 2, 3])
    end

    it 'raises ArgumentError when given invalid shape array', :aggregate_failures do
      expect { described_class.check_convert_target_value_array(Numo::DFloat.cast(arr1d)) }.not_to raise_error
      expect { described_class.check_convert_target_value_array(Numo::DFloat.cast(arr2d)) }.not_to raise_error
      expect { described_class.check_convert_target_value_array(Numo::DFloat.cast(arr3d)) }.to raise_error(ArgumentError)
    end
  end

  describe '#check_sample_size' do
    let(:x) { Numo::DFloat[[1, 2], [3, 4], [5, 6]] }

    it 'raises ArgumentError when given invalid number of samples', :aggregate_failures do
      expect { described_class.check_sample_size(x, Numo::Int32[1, 2]) }.to raise_error(ArgumentError)
      expect { described_class.check_sample_size(x, Numo::DFloat[1, 2]) }.to raise_error(ArgumentError)
    end
  end
end
