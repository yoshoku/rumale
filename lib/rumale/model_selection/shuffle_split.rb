# frozen_string_literal: true

require 'rumale/base/splitter'

module Rumale
  module ModelSelection
    # ShuffleSplit is a class that generates the set of data indices for random permutation cross-validation.
    #
    # @example
    #   ss = Rumale::ModelSelection::ShuffleSplit.new(n_splits: 3, test_size: 0.2, random_seed: 1)
    #   ss.split(samples, labels).each do |train_ids, test_ids|
    #     train_samples = samples[train_ids, true]
    #     test_samples = samples[test_ids, true]
    #     ...
    #   end
    #
    class ShuffleSplit
      include Base::Splitter

      # Return the number of folds.
      # @return [Integer]
      attr_reader :n_splits

      # Return the random generator for shuffling the dataset.
      # @return [Random]
      attr_reader :rng

      # Create a new data splitter for random permutation cross validation.
      #
      # @param n_splits [Integer] The number of folds.
      # @param test_size [Float] The ratio of number of samples for test data.
      # @param train_size [Float] The ratio of number of samples for train data.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_splits: 3, test_size: 0.1, train_size: nil, random_seed: nil)
        check_params_numeric(n_splits: n_splits, test_size: test_size)
        check_params_numeric_or_nil(train_size: train_size, random_seed: random_seed)
        check_params_positive(n_splits: n_splits)
        check_params_positive(test_size: test_size)
        check_params_positive(train_size: train_size) unless train_size.nil?
        @n_splits = n_splits
        @test_size = test_size
        @train_size = train_size
        @random_seed = random_seed
        @random_seed ||= srand
        @rng = Random.new(@random_seed)
      end

      # Generate data indices for random permutation cross validation.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The dataset to be used to generate data indices for random permutation cross validation.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(x, _y = nil)
        x = check_convert_sample_array(x)
        # Initialize and check some variables.
        n_samples = x.shape[0]
        n_test_samples = (@test_size * n_samples).to_i
        n_train_samples = @train_size.nil? ? n_samples - n_test_samples : (@train_size * n_samples).to_i
        unless @n_splits.between?(1, n_samples)
          raise ArgumentError,
                'The value of n_splits must be not less than 1 and not more than the number of samples.'
        end
        unless n_test_samples.between?(1, n_samples)
          raise RangeError,
                'The number of sample in test split must be not less than 1 and not more than the number of samples.'
        end
        unless n_train_samples.between?(1, n_samples)
          raise RangeError,
                'The number of sample in train split must be not less than 1 and not more than the number of samples.'
        end
        if (n_test_samples + n_train_samples) > n_samples
          raise RangeError,
                'The total number of samples in test split and train split must be not more than the number of samples.'
        end
        sub_rng = @rng.dup
        # Returns array consisting of the training and testing ids for each fold.
        dataset_ids = Array(0...n_samples)
        Array.new(@n_splits) do
          test_ids = dataset_ids.sample(n_test_samples, random: sub_rng)
          train_ids = if @train_size.nil?
                        dataset_ids - test_ids
                      else
                        (dataset_ids - test_ids).sample(n_train_samples, random: sub_rng)
                      end
          [train_ids, test_ids]
        end
      end
    end
  end
end
