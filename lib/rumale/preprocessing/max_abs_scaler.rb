# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Normalize samples by scaling each feature with its maximum absolute value.
    #
    # @example
    #   normalizer = Rumale::Preprocessing::MaxAbsScaler.new
    #   new_training_samples = normalizer.fit_transform(training_samples)
    #   new_testing_samples = normalizer.transform(testing_samples)
    class MaxAbsScaler
      include Base::BaseEstimator
      include Base::Transformer

      # Return the vector consists of the maximum absolute value for each feature.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :max_abs_vec

      # Creates a new normalizer for scaling each feature with its maximum absolute value.
      def initialize
        @params = {}
        @max_abs_vec = nil
      end

      # Calculate the minimum and maximum value of each feature for scaling.
      #
      # @overload fit(x) -> MaxAbsScaler
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate maximum absolute value for each feature.
      # @return [MaxAbsScaler]
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        @max_abs_vec = x.abs.max(0)
        self
      end

      # Calculate the maximum absolute value for each feature, and then normalize samples.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate maximum absolute value for each feature.
      # @return [Numo::DFloat] The scaled samples.
      def fit_transform(x, _y = nil)
        x = check_convert_sample_array(x)
        fit(x).transform(x)
      end

      # Perform scaling the given samples with maximum absolute value for each feature.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be scaled.
      # @return [Numo::DFloat] The scaled samples.
      def transform(x)
        x = check_convert_sample_array(x)
        x / @max_abs_vec
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about MaxAbsScaler.
      def marshal_dump
        { params: @params,
          max_abs_vec: @max_abs_vec }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @max_abs_vec = obj[:max_abs_vec]
        nil
      end
    end
  end
end
