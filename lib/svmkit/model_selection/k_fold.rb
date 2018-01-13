require 'svmkit/base/splitter'

module SVMKit
  # This module consists of the classes for model validation techniques.
  module ModelSelection
    # KFold is a class that generates the set of data indices for K-fold cross-validation.
    #
    # @example
    #   kf = SVMKit::ModelSelection::KFold.new(n_splits: 3, shuffle: true, random_seed: 1)
    #   kf.split(samples, labels).each do |train_ids, test_ids|
    #     train_samples = samples[train_ids, true]
    #     test_samples = samples[test_ids, true]
    #     ...
    #   end
    #
    class KFold
      include Base::Splitter

      # Return the proportion of the test set to the dataset.
      # @return [Boolean]
      attr_reader :shuffle

      # Return the random generator for shuffling the dataset.
      # @return [Random]
      attr_reader :rng

      # Create a new data splitter for K-fold cross validation.
      #
      # @param n_splits [Integer] The number of folds.
      # @param shuffle [Boolean] The flag indicating whether to shuffle the dataset.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_splits: 3, shuffle: false, random_seed: nil)
        @n_splits = n_splits
        @shuffle = shuffle
        @random_seed = random_seed
        @random_seed ||= srand
        @rng = Random.new(@random_seed)
      end

      # Generate data indices for K-fold cross validation.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The dataset to be used to generate data indices for K-fold cross validation.
      # @param y [Numo::Int32] (shape: [n_samples])
      #   The labels to be used to generate data indices for stratified K-fold cross validation.
      #   This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(x, y) # rubocop:disable Lint/UnusedMethodArgument
        # Initialize and check some variables.
        n_samples, = x.shape
        unless @n_splits.between?(2, n_samples)
          raise ArgumentError,
                'The value of n_splits must be not less than 2 and not more than the number of samples.'
        end
        # Splits dataset ids to each fold.
        dataset_ids = [*0...n_samples]
        dataset_ids.shuffle!(random: @rng) if @shuffle
        fold_sets = Array.new(@n_splits) do |n|
          n_fold_samples = n_samples / @n_splits
          n_fold_samples += 1 if n < n_samples % @n_splits
          dataset_ids.shift(n_fold_samples)
        end
        # Returns array consisting of the training and testing ids for each fold.
        Array.new(@n_splits) do |n|
          train_ids = fold_sets.select.with_index { |_, id| id != n }.flatten
          test_ids = fold_sets[n]
          [train_ids, test_ids]
        end
      end
    end
  end
end
