# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  # This module consists of the classes that perform preprocessings.
  module Preprocessing
    # Normalize samples by centering and scaling to unit variance.
    #
    # @example
    #   normalizer = Rumale::Preprocessing::StandardScaler.new
    #   new_training_samples = normalizer.fit_transform(training_samples)
    #   new_testing_samples = normalizer.transform(testing_samples)
    class StandardScaler
      include Base::BaseEstimator
      include Base::Transformer

      # Return the vector consists of the mean value for each feature.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :mean_vec

      # Return the vector consists of the standard deviation for each feature.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :std_vec

      # Create a new normalizer for centering and scaling to unit variance.
      def initialize
        @params = {}
        @mean_vec = nil
        @std_vec = nil
      end

      # Calculate the mean value and standard deviation of each feature for scaling.
      #
      # @overload fit(x) -> StandardScaler
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The samples to calculate the mean values and standard deviations.
      # @return [StandardScaler]
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        @mean_vec = x.mean(0)
        @std_vec = x.stddev(0)
        self
      end

      # Calculate the mean values and standard deviations, and then normalize samples using them.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The samples to calculate the mean values and standard deviations.
      # @return [Numo::DFloat] The scaled samples.
      def fit_transform(x, _y = nil)
        x = check_convert_sample_array(x)
        fit(x).transform(x)
      end

      # Perform standardization the given samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be scaled.
      # @return [Numo::DFloat] The scaled samples.
      def transform(x)
        x = check_convert_sample_array(x)
        n_samples, = x.shape
        (x - @mean_vec.tile(n_samples, 1)) / @std_vec.tile(n_samples, 1)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about StandardScaler.
      def marshal_dump
        { mean_vec: @mean_vec,
          std_vec: @std_vec }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @mean_vec = obj[:mean_vec]
        @std_vec = obj[:std_vec]
        nil
      end
    end
  end
end
