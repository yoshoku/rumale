# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/validation'

module Rumale
  module Preprocessing
    # Normalize samples by scaling each feature with its maximum absolute value.
    #
    # @example
    #   require 'rumale/preprocessing/max_abs_scaler'
    #
    #   normalizer = Rumale::Preprocessing::MaxAbsScaler.new
    #   new_training_samples = normalizer.fit_transform(training_samples)
    #   new_testing_samples = normalizer.transform(testing_samples)
    class MaxAbsScaler < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Return the vector consists of the maximum absolute value for each feature.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :max_abs_vec

      # Creates a new normalizer for scaling each feature with its maximum absolute value.
      def initialize # rubocop:disable Lint/UselessMethodDefinition
        super
      end

      # Calculate the minimum and maximum value of each feature for scaling.
      #
      # @overload fit(x) -> MaxAbsScaler
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate maximum absolute value for each feature.
      # @return [MaxAbsScaler]
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

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
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit(x).transform(x)
      end

      # Perform scaling the given samples with maximum absolute value for each feature.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be scaled.
      # @return [Numo::DFloat] The scaled samples.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        x / @max_abs_vec
      end
    end
  end
end
