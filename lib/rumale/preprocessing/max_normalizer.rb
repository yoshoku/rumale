# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Normalize samples with the maximum of the absolute values.
    #
    # @example
    #   normalizer = Rumale::Preprocessing::MaxNormalizer.new
    #   new_samples = normalizer.fit_transform(samples)
    class MaxNormalizer
      include Base::BaseEstimator
      include Base::Transformer

      # Return the vector consists of the maximum norm for each sample.
      # @return [Numo::DFloat] (shape: [n_samples])
      attr_reader :norm_vec # :nodoc:

      # Create a new normalizer for normaliing to max-norm.
      def initialize
        @params = {}
        @norm_vec = nil
      end

      # Calculate the maximum norms of each sample.
      #
      # @overload fit(x) -> MaxNormalizer
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the maximum norms.
      # @return [MaxNormalizer]
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        @norm_vec = x.abs.max(1)
        @norm_vec[@norm_vec.eq(0)] = 1
        self
      end

      # Calculate the maximums norm of each sample, and then normalize samples with the norms.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate maximum norms.
      # @return [Numo::DFloat] The normalized samples.
      def fit_transform(x, _y = nil)
        x = check_convert_sample_array(x)
        fit(x)
        x / @norm_vec.expand_dims(1)
      end

      # Calculate the maximum norms of each sample, and then normalize samples with the norms.
      # This method calls the fit_transform method. This method exists for the Pipeline class.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate maximum norms.
      # @return [Numo::DFloat] The normalized samples.
      def transform(x)
        fit_transform(x)
      end
    end
  end
end
