require 'svmkit/base/splitter'

module SVMKit
  module ModelSelection
    # StratifiedKFold is a class that generates the set of data indices for K-fold cross-validation.
    # The proportion of the number of samples in each class will be almost equal for each fold.
    #
    # @example
    #   kf = SVMKit::ModelSelection::StratifiedKFold.new(n_splits: 3, shuffle: true, random_seed: 1)
    #   kf.split(samples, labels).each do |train_ids, test_ids|
    #     train_samples = samples[train_ids, true]
    #     test_samples = samples[test_ids, true]
    #     ...
    #   end
    #
    class StratifiedKFold
      include Base::Splitter

      # Return the flag indicating whether to shuffle the dataset.
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

      # Generate data indices for stratified K-fold cross validation.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The dataset to be used to generate data indices for stratified K-fold cross validation.
      #   This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      # @param y [Numo::Int32] (shape: [n_samples])
      #   The labels to be used to generate data indices for stratified K-fold cross validation.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(x, y) # rubocop:disable Lint/UnusedMethodArgument
        # Check the number of samples in each class.
        unless y.bincount.to_a.all? { |n_samples| @n_splits.between?(2, n_samples) }
          raise ArgumentError,
                'The value of n_splits must be not less than 2 and not more than the number of samples in each class.'
        end
        # Splits dataset ids of each class to each fold.
        fold_sets_each_class = y.to_a.uniq.map { |label| fold_sets(y, label) }
        # Returns array consisting of the training and testing ids for each fold.
        Array.new(@n_splits) { |fold_id| train_test_sets(fold_sets_each_class, fold_id) }
      end

      private

      def fold_sets(y, label)
        sample_ids = y.eq(label).where.to_a
        sample_ids.shuffle!(random: @rng) if @shuffle
        n_samples = sample_ids.size
        Array.new(@n_splits) do |n|
          n_fold_samples = n_samples / @n_splits
          n_fold_samples += 1 if n < n_samples % @n_splits
          sample_ids.shift(n_fold_samples)
        end
      end

      def train_test_sets(fold_sets_each_class, fold_id)
        train_test_sets_each_class = fold_sets_each_class.map do |folds|
          folds.partition.with_index { |_, id| id != fold_id }.map(&:flatten)
        end
        train_test_sets_each_class.transpose.map(&:flatten)
      end
    end
  end
end
