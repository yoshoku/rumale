# frozen_string_literal: true

require 'rumale/base/splitter'

module Rumale
  module ModelSelection
    # TimeSeriesSplit is a class that generates the set of data indices for time series cross-validation.
    # It is assumed that the dataset given are already ordered by time information.
    #
    # @example
    #   cv = Rumale::ModelSelection::TimeSeriesSplit.new(n_splits: 5)
    #   x = Numo::DFloat.new(6, 2).rand
    #   cv.split(x, nil).each do |train_ids, test_ids|
    #     puts '---'
    #     pp train_ids
    #     pp test_ids
    #   end
    #
    #   # ---
    #   # [0]
    #   # [1]
    #   # ---
    #   # [0, 1]
    #   # [2]
    #   # ---
    #   # [0, 1, 2]
    #   # [3]
    #   # ---
    #   # [0, 1, 2, 3]
    #   # [4]
    #   # ---
    #   # [0, 1, 2, 3, 4]
    #   # [5]
    #
    class TimeSeriesSplit
      include Base::Splitter

      # Return the number of splits.
      # @return [Integer]
      attr_reader :n_splits

      # Return the maximum number of training samples in a split.
      # @return [Integer/Nil]
      attr_reader :max_train_size

      # Create a new data splitter for time series cross-validation.
      #
      # @param n_splits [Integer] The number of splits.
      # @param max_train_size [Integer/Nil] The maximum number of training samples in a split.
      def initialize(n_splits: 5, max_train_size: nil)
        check_params_numeric(n_splits: n_splits)
        check_params_numeric_or_nil(max_train_size: max_train_size)
        @n_splits = n_splits
        @max_train_size = max_train_size
      end

      # Generate data indices for time series cross-validation.
      #
      # @overload split(x, y) -> Array
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #     The dataset to be used to generate data indices for time series cross-validation.
      #     It is expected that the data will be ordered by time information.
      #   @param y [Numo::Int32] (shape: [n_samples])
      #     This argument exists to unify the interface between the K-fold methods, it is not used in the method.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def split(x, _y)
        x = check_convert_sample_array(x)

        n_samples = x.shape[0]
        unless (@n_splits + 1).between?(2, n_samples)
          raise ArgumentError,
                'The number of folds (n_splits + 1) must be not less than 2 and not more than the number of samples.'
        end

        test_size = n_samples / (@n_splits + 1)
        offset = test_size + n_samples % (@n_splits + 1)

        Array.new(@n_splits) do |n|
          start = offset * (n + 1)
          train_ids = if !@max_train_size.nil? && @max_train_size < test_size
                        Array((start - @max_train_size)...start)
                      else
                        Array(0...start)
                      end
          test_ids = Array(start...(start + test_size))
          [train_ids, test_ids]
        end
      end
    end
  end
end
