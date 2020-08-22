# frozen_string_literal: true

require 'rumale/base/splitter'

module Rumale
  module ModelSelection
    # StratifiedShuffleSplit is a class that generates the set of data indices for random permutation cross-validation.
    # The proportion of the number of samples in each class will be almost equal for each fold.
    #
    # @example
    #   ss = Rumale::ModelSelection::StratifiedShuffleSplit.new(n_splits: 3, test_size: 0.2, random_seed: 1)
    #   ss.split(samples, labels).each do |train_ids, test_ids|
    #     train_samples = samples[train_ids, true]
    #     test_samples = samples[test_ids, true]
    #     ...
    #   end
    #
    class StratifiedShuffleSplit
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

      # Generate data indices for stratified random permutation cross validation.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The dataset to be used to generate data indices for stratified random permutation cross validation.
      #   This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      # @param y [Numo::Int32] (shape: [n_samples])
      #   The labels to be used to generate data indices for stratified random permutation cross validation.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)
        # Initialize and check some variables.
        train_sz = @train_size.nil? ? 1.0 - @test_size : @train_size
        sub_rng = @rng.dup
        # Check the number of samples in each class.
        unless valid_n_splits?(y)
          raise ArgumentError,
                'The value of n_splits must be not less than 1 and not more than the number of samples in each class.'
        end
        unless enough_data_size_each_class?(y, @test_size, 'test')
          raise RangeError,
                'The number of samples in test split must be not less than 1 and not more than the number of samples in each class.'
        end
        unless enough_data_size_each_class?(y, train_sz, 'train')
          raise RangeError,
                'The number of samples in train split must be not less than 1 and not more than the number of samples in each class.'
        end
        unless enough_data_size_each_class?(y, train_sz + @test_size, 'train')
          raise RangeError,
                'The total number of samples in test split and train split must be not more than the number of samples in each class.'
        end
        # Returns array consisting of the training and testing ids for each fold.
        sample_ids_each_class = y.to_a.uniq.map { |label| y.eq(label).where.to_a }
        Array.new(@n_splits) do
          train_ids = []
          test_ids = []
          sample_ids_each_class.each do |sample_ids|
            n_samples = sample_ids.size
            n_test_samples = (@test_size * n_samples).ceil.to_i
            test_ids += sample_ids.sample(n_test_samples, random: sub_rng)
            train_ids += if @train_size.nil?
                           sample_ids - test_ids
                         else
                           n_train_samples = (train_sz * n_samples).floor.to_i
                           (sample_ids - test_ids).sample(n_train_samples, random: sub_rng)
                         end
          end
          [train_ids, test_ids]
        end
      end

      private

      def valid_n_splits?(y)
        y.to_a.uniq.map { |label| y.eq(label).where.size }.all? { |n_samples| @n_splits.between?(1, n_samples) }
      end

      def enough_data_size_each_class?(y, data_size, data_type)
        y.to_a.uniq.map { |label| y.eq(label).where.size }.all? do |n_samples|
          if data_type == 'test'
            (data_size * n_samples).ceil.to_i.between?(1, n_samples)
          else
            (data_size * n_samples).floor.to_i.between?(1, n_samples)
          end
        end
      end
    end
  end
end
