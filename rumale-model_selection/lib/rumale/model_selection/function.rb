# frozen_string_literal: true

require 'numo/narray'

require 'rumale/model_selection/shuffle_split'
require 'rumale/model_selection/stratified_shuffle_split'

module Rumale
  # This module consists of the classes for model validation techniques.
  module ModelSelection
    module_function

    # Split randomly data set into test and train data.
    #
    # @example
    #   require 'rumale/model_selection/function'
    #
    #   x_train, x_test, y_train, y_test = Rumale::ModelSelection.train_test_split(x, y, test_size: 0.2, stratify: true, random_seed: 1)
    #
    # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The dataset to be used to generate data indices.
    # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used to generate data indices for stratified random permutation.
    #   If stratify = false, this parameter is ignored.
    # @param test_size [Float] The ratio of number of samples for test data.
    # @param train_size [Float] The ratio of number of samples for train data.
    #   If nil is given, it sets to 1 - test_size.
    # @param stratify [Boolean] The flag indicating whether to perform stratify split.
    # @param random_seed [Integer] The seed value using to initialize the random generator.
    # @return [Array<Numo::NArray>] The set of training and testing data.
    def train_test_split(x, y = nil, test_size: 0.1, train_size: nil, stratify: false, random_seed: nil)
      splitter = if stratify
                   ::Rumale::ModelSelection::StratifiedShuffleSplit.new(
                     n_splits: 1, test_size: test_size, train_size: train_size, random_seed: random_seed
                   )
                 else
                   ::Rumale::ModelSelection::ShuffleSplit.new(
                     n_splits: 1, test_size: test_size, train_size: train_size, random_seed: random_seed
                   )
                 end
      train_ids, test_ids = splitter.split(x, y).first
      x_train = x[train_ids, true].dup
      y_train = y[train_ids].dup
      x_test = x[test_ids, true].dup
      y_test = y[test_ids].dup
      [x_train, x_test, y_train, y_test]
    end
  end
end
