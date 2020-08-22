# frozen_string_literal: true

require 'rumale/base/splitter'

module Rumale
  module ModelSelection
    # GroupShuffleSplit is a class that generates the set of data indices
    # for random permutation cross-validation by randomly selecting group labels.
    #
    # @example
    #   cv = Rumale::ModelSelection::GroupShuffleSplit.new(n_splits: 2, test_size: 0.2, random_seed: 1)
    #   x = Numo::DFloat.new(8, 2).rand
    #   groups = Numo::Int32[1, 1, 1, 2, 2, 3, 3, 3]
    #   cv.split(x, nil, groups).each do |train_ids, test_ids|
    #     puts '---'
    #     pp train_ids
    #     pp test_ids
    #   end
    #
    #   # ---
    #   # [0, 1, 2, 5, 6, 7]
    #   # [3, 4]
    #   # ---
    #   # [3, 4, 5, 6, 7]
    #   # [0, 1, 2]
    #
    class GroupShuffleSplit
      include Base::Splitter

      # Return the number of folds.
      # @return [Integer]
      attr_reader :n_splits

      # Return the random generator for shuffling the dataset.
      # @return [Random]
      attr_reader :rng

      # Create a new data splitter for random permutation cross validation with given group labels.
      #
      # @param n_splits [Integer] The number of folds.
      # @param test_size [Float] The ratio of number of groups for test data.
      # @param train_size [Float/Nil] The ratio of number of groups for train data.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_splits: 5, test_size: 0.2, train_size: nil, random_seed: nil)
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

      # Generate train and test data indices by randomly selecting group labels.
      #
      # @overload split(x, y, groups) -> Array
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #     The dataset to be used to generate data indices for random permutation cross validation.
      #   @param y [Numo::Int32] (shape: [n_samples])
      #     This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      #   @param groups [Numo::Int32] (shape: [n_samples])
      #     The group labels to be used to generate data indices for random permutation cross validation.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(x, _y, groups)
        x = check_convert_sample_array(x)
        groups = check_convert_label_array(groups)
        check_sample_label_size(x, groups)

        classes = groups.to_a.uniq.sort
        n_groups = classes.size
        n_test_groups = (@test_size * n_groups).ceil.to_i
        n_train_groups = @train_size.nil? ? n_groups - n_test_groups : (@train_size * n_groups).floor.to_i

        unless n_test_groups.between?(1, n_groups)
          raise RangeError,
                'The number of groups in test split must be not less than 1 and not more than the number of groups.'
        end
        unless n_train_groups.between?(1, n_groups)
          raise RangeError,
                'The number of groups in train split must be not less than 1 and not more than the number of groups.'
        end
        if (n_test_groups + n_train_groups) > n_groups
          raise RangeError,
                'The total number of groups in test split and train split must be not more than the number of groups.'
        end

        sub_rng = @rng.dup

        Array.new(@n_splits) do
          test_group_ids = classes.sample(n_test_groups, random: sub_rng)
          train_group_ids = if @train_size.nil?
                              classes - test_group_ids
                            else
                              (classes - test_group_ids).sample(n_train_groups, random: sub_rng)
                            end
          test_ids = in1d(groups, test_group_ids).where.to_a
          train_ids = in1d(groups, train_group_ids).where.to_a
          [train_ids, test_ids]
        end
      end

      private

      def in1d(a, b)
        res = Numo::Bit.zeros(a.shape[0])
        b.each { |v| res |= a.eq(v) }
        res
      end
    end
  end
end
