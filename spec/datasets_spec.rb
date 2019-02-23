# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::Dataset do
  let(:labels) { Numo::Int32.asarray([1, 2, 2, 1, 1, 0]) }
  let(:target_variables) { Numo::DFloat.asarray([1.2, 2.0, 2.3, 1.0, 1.1, 0.64]) }
  let(:mult_target_vals) { Numo::DFloat.asarray([[1.2, 2.0], [2.3, 1.0], [1.1, 0.64], [2.1, 1.9], [0.0, 1.7], [8.7, 4.1]]) }

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

  it 'loads libsvm .t file containing double features for regression task.' do
    m, t = described_class.load_libsvm_file(__dir__ + '/test_dbl.t')
    expect(m).to eq(matrix_dbl)
    expect(m.class).to eq(Numo::DFloat)
    expect(t).to eq(target_variables)
    expect(t.class).to eq(Numo::DFloat)
  end

  it 'loads libsvm .t file containing integer features for classification task.' do
    m, l = described_class.load_libsvm_file(__dir__ + '/test_int.t', dtype: Numo::Int32)
    expect(m).to eq(matrix_int)
    expect(m.class).to eq(Numo::Int32)
    expect(l).to eq(labels)
    expect(l.class).to eq(Numo::Int32)
  end

  it 'dumps and loads double features wit multi-target variables.' do
    described_class.dump_libsvm_file(matrix_dbl, mult_target_vals, __dir__ + '/dump_mult_dbl.t')
    m, t = described_class.load_libsvm_file(__dir__ + '/dump_mult_dbl.t')
    expect(m).to eq(matrix_dbl)
    expect(m.class).to eq(Numo::DFloat)
    expect(t).to eq(mult_target_vals)
    expect(t.class).to eq(Numo::DFloat)
  end

  it 'loads libsvm .t file with zero-based indexing.' do
    m, = described_class.load_libsvm_file(__dir__ + '/test_zb.t', zero_based: true)
    expect(m).to eq(matrix_dbl)
  end

  it 'dumps double features with target variables.' do
    described_class.dump_libsvm_file(matrix_dbl, target_variables, __dir__ + '/dump_dbl.t')
    m, t = described_class.load_libsvm_file(__dir__ + '/dump_dbl.t')
    expect(m).to eq(matrix_dbl)
    expect(t).to eq(target_variables)
  end

  it 'dumps integer features with labels.' do
    described_class.dump_libsvm_file(matrix_int, labels, __dir__ + '/dump_int.t')
    m, l = described_class.load_libsvm_file(__dir__ + '/dump_int.t')
    expect(m).to eq(matrix_int)
    expect(l).to eq(labels)
  end

  it 'dumps features with zero-based indexing.' do
    described_class.dump_libsvm_file(matrix_dbl, labels, __dir__ + '/dump_zb.t', zero_based: true)
    m, l = described_class.load_libsvm_file(__dir__ + '/dump_zb.t', zero_based: true)
    expect(m).to eq(matrix_dbl)
    expect(l).to eq(labels)
  end
end
