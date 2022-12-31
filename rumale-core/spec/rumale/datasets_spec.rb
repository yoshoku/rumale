# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Dataset do
  let(:labels) { Numo::Int32.asarray([1, 2, 2, 1, 1, 0]) }
  let(:target_variables) { Numo::DFloat.asarray([1.2, 2.0, 2.3, 1.0, 1.1, 0.64]) }
  let(:mult_target_vals) { Numo::DFloat.asarray([[1.2, 2.0], [2.3, 1.0], [1.1, 0.64], [2.1, 1.9], [0.0, 1.7], [8.7, 4.1]]) }
  let(:file_fixture_path) { "#{__dir__}/../fixtures/files" }

  let(:matrix_int) do
    Numo::Int32.asarray([
                          [5, 3, 0, 8],
                          [3, 1, 2, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 2],
                          [0, 4, 0, 5]
                        ])
  end

  let(:matrix_dbl) do
    Numo::DFloat.asarray([
                           [5.0, 3.1, 0.0, 8.40],
                           [3.2, 1.2, 2.5, 0.00],
                           [0.0, 0.0, 1.3, 0.00],
                           [0.0, 0.0, 0.0, 0.00],
                           [0.1, 0.0, 0.0, 2.56],
                           [0.0, 4.8, 0.0, 5.12]
                         ])
  end

  describe '#load_libsvm_file' do
    it 'loads libsvm .t file containing double features for regression task', :aggregate_failures do
      m, t = described_class.load_libsvm_file("#{file_fixture_path}/test_dbl.t")
      expect(m).to eq(matrix_dbl)
      expect(m).to be_a(Numo::DFloat)
      expect(t).to eq(target_variables)
      expect(t).to be_a(Numo::DFloat)
    end

    it 'loads libsvm .t file containing integer features for classification task', :aggregate_failures do
      m, l = described_class.load_libsvm_file("#{file_fixture_path}/test_int.t", dtype: Numo::Int32)
      expect(m).to eq(matrix_int)
      expect(m).to be_a(Numo::Int32)
      expect(l).to eq(labels)
      expect(l).to be_a(Numo::Int32)
    end

    it 'dumps and loads double features with multi-target variables', :aggregate_failures do
      described_class.dump_libsvm_file(matrix_dbl, mult_target_vals, "#{file_fixture_path}/dump_mult_dbl.t")
      m, t = described_class.load_libsvm_file("#{file_fixture_path}/dump_mult_dbl.t")
      expect(m).to eq(matrix_dbl)
      expect(m).to be_a(Numo::DFloat)
      expect(t).to eq(mult_target_vals)
      expect(t).to be_a(Numo::DFloat)
    end

    it 'loads libsvm .t file with zero-based indexing', :aggregate_failures do
      m, = described_class.load_libsvm_file("#{file_fixture_path}/test_zb.t", zero_based: true)
      expect(m).to eq(matrix_dbl)
    end

    it 'lodas libsvm .t file with the number of features', :aggregate_failures do
      m, = described_class.load_libsvm_file("#{file_fixture_path}/test_dbl.t", n_features: 6)
      expect(m.shape[1]).to eq(6)
      expect(m).to eq(matrix_dbl.concatenate(Numo::DFloat.zeros(6, 2), axis: 1))
      m, = described_class.load_libsvm_file("#{file_fixture_path}/test_dbl.t", n_features: 2)
      expect(m.shape[1]).to eq(matrix_dbl.shape[1])
      m, = described_class.load_libsvm_file("#{file_fixture_path}/test_zb.t", zero_based: true, n_features: 6)
      expect(m.shape[1]).to eq(6)
      expect(m).to eq(matrix_dbl.concatenate(Numo::DFloat.zeros(6, 2), axis: 1))
      m, = described_class.load_libsvm_file("#{file_fixture_path}/test_zb.t", zero_based: true, n_features: 2)
      expect(m.shape[1]).to eq(matrix_dbl.shape[1])
    end
  end

  describe '#dump_libsvm_file' do
    it 'dumps double features with target variables', :aggregate_failures do
      described_class.dump_libsvm_file(matrix_dbl, target_variables, "#{file_fixture_path}/dump_dbl.t")
      m, t = described_class.load_libsvm_file("#{file_fixture_path}/dump_dbl.t")
      expect(m).to eq(matrix_dbl)
      expect(t).to eq(target_variables)
    end

    it 'dumps integer features with labels', :aggregate_failures do
      described_class.dump_libsvm_file(matrix_int, labels, "#{file_fixture_path}/dump_int.t")
      m, l = described_class.load_libsvm_file("#{file_fixture_path}/dump_int.t")
      expect(m).to eq(matrix_int)
      expect(l).to eq(labels)
    end

    it 'dumps features with zero-based indexing', :aggregate_failures do
      described_class.dump_libsvm_file(matrix_dbl, labels, "#{file_fixture_path}/dump_zb.t", zero_based: true)
      m, l = described_class.load_libsvm_file("#{file_fixture_path}/dump_zb.t", zero_based: true)
      expect(m).to eq(matrix_dbl)
      expect(l).to eq(labels)
    end
  end

  describe '#make_circles' do
    it 'generates two circles data', :aggregate_failures do
      x, y = described_class.make_circles(100, noise: 0.05)
      expect(x).to be_a(Numo::DFloat)
      expect(x.shape[0]).to eq(100)
      expect(x.shape[1]).to eq(2)
      expect(y).to be_a(Numo::Int32)
      expect(y.shape[0]).to eq(100)
      expect(y.shape[1]).to be_nil
      expect(y.eq(0).count).to eq(50)
      expect(y.eq(1).count).to eq(50)
    end
  end

  describe '#make_moons' do
    it 'generates two moons data', :aggregate_failures do
      x, y = described_class.make_moons(100, noise: 0.05)
      expect(x).to be_a(Numo::DFloat)
      expect(x.shape[0]).to eq(100)
      expect(x.shape[1]).to eq(2)
      expect(y).to be_a(Numo::Int32)
      expect(y.shape[0]).to eq(100)
      expect(y.shape[1]).to be_nil
      expect(y.eq(0).count).to eq(50)
      expect(y.eq(1).count).to eq(50)
    end
  end

  describe '#make_blobs' do
    it 'generates Gaussian blobs', :aggregate_failures do
      x, y = described_class.make_blobs(100)
      expect(x).to be_a(Numo::DFloat)
      expect(x.shape[0]).to eq(100)
      expect(x.shape[1]).to eq(2)
      expect(y).to be_a(Numo::Int32)
      expect(y.shape[0]).to eq(100)
      expect(y.shape[1]).to be_nil
      expect(y.eq(0).count).to eq(34)
      expect(y.eq(1).count).to eq(33)
      expect(y.eq(2).count).to eq(33)
    end

    it 'generates Gaussian blobs along with given centers', :aggregate_failures do
      centers = Numo::DFloat[[-20, -20], [20, 20]]
      x, y = described_class.make_blobs(100, 3, centers: centers, cluster_std: 0.05)
      expect(x).to be_a(Numo::DFloat)
      expect(x.shape[0]).to eq(100)
      expect(x.shape[1]).to eq(2)
      expect(y).to be_a(Numo::Int32)
      expect(y.shape[0]).to eq(100)
      expect(y.shape[1]).to be_nil
      expect(y.eq(0).count).to eq(50)
      expect(y.eq(1).count).to eq(50)
      expect(Math.sqrt(((x[y.eq(0), true].mean(0) - centers[0, true])**2).sum)).to be < 0.1
      expect(Math.sqrt(((x[y.eq(1), true].mean(0) - centers[1, true])**2).sum)).to be < 0.1
    end
  end
end
