# frozen_string_literal: true

require 'rumale/base/splitter'
require 'rumale/preprocessing/label_encoder'

module Rumale
  module ModelSelection
    # GroupKFold is a class that generates the set of data indices for K-fold cross-validation.
    # The data points belonging to the same group do not be split into different folds.
    # The number of groups should be greater than or equal to the number of splits.
    #
    # @example
    #   require 'rumale/model_selection/group_k_fold'
    #
    #   cv = Rumale::ModelSelection::GroupKFold.new(n_splits: 3)
    #   x = Numo::DFloat.new(8, 2).rand
    #   groups = Numo::Int32[1, 1, 1, 2, 2, 3, 3, 3]
    #   cv.split(x, nil, groups).each do |train_ids, test_ids|
    #     puts '---'
    #     pp train_ids
    #     pp test_ids
    #   end
    #
    #   # ---
    #   # [0, 1, 2, 3, 4]
    #   # [5, 6, 7]
    #   # ---
    #   # [3, 4, 5, 6, 7]
    #   # [0, 1, 2]
    #   # ---
    #   # [0, 1, 2, 5, 6, 7]
    #   # [3, 4]
    #
    class GroupKFold
      include ::Rumale::Base::Splitter

      # Return the number of folds.
      # @return [Integer]
      attr_reader :n_splits

      # Create a new data splitter for grouped K-fold cross validation.
      #
      # @param n_splits [Integer] The number of folds.
      def initialize(n_splits: 5)
        @n_splits = n_splits
      end

      # Generate data indices for grouped K-fold cross validation.
      #
      # @overload split(x, y, groups) -> Array
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #     The dataset to be used to generate data indices for grouped K-fold cross validation.
      #   @param y [Numo::Int32] (shape: [n_samples])
      #     This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      #   @param groups [Numo::Int32] (shape: [n_samples])
      #     The group labels to be used to generate data indices for grouped K-fold cross validation.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(x, _y, groups)
        encoder = ::Rumale::Preprocessing::LabelEncoder.new
        groups = encoder.fit_transform(groups)
        n_groups = encoder.classes.size

        if n_groups < @n_splits
          raise ArgumentError,
                'The number of groups should be greater than or equal to the number of splits.'
        end

        n_samples_per_group = groups.bincount
        group_ids = n_samples_per_group.sort_index.reverse
        n_samples_per_group = n_samples_per_group[group_ids]

        n_samples_per_fold = Numo::Int32.zeros(@n_splits)
        group_to_fold = Numo::Int32.zeros(n_groups)

        n_samples_per_group.each_with_index do |weight, id|
          min_sample_fold_id = n_samples_per_fold.min_index
          n_samples_per_fold[min_sample_fold_id] += weight
          group_to_fold[group_ids[id]] = min_sample_fold_id
        end

        n_samples = x.shape[0]
        sample_ids = Array(0...n_samples)
        fold_ids = group_to_fold[groups]

        Array.new(@n_splits) do |fid|
          test_ids = fold_ids.eq(fid).where.to_a
          train_ids = sample_ids - test_ids
          [train_ids, test_ids]
        end
      end
    end
  end
end
