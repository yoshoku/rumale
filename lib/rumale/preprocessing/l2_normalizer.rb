# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  # This module consists of the classes that perform preprocessings.
  module Preprocessing
    # Normalize samples to unit L2-norm.
    #
    # @example
    #   normalizer = Rumale::Preprocessing::L2Normalizer.new
    #   new_samples = normalizer.fit_transform(samples)
    class L2Normalizer
      include Base::BaseEstimator
      include Base::Transformer

      # Return the vector consists of L2-norm for each sample.
      # @return [Numo::DFloat] (shape: [n_samples])
      attr_reader :norm_vec # :nodoc:

      # Create a new normalizer for normaliing to unit L2-norm.
      def initialize
        @params = {}
        @norm_vec = nil
      end

      # Calculate L2-norms of each sample.
      #
      # @overload fit(x) -> L2Normalizer
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate L2-norms.
      # @return [L2Normalizer]
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        @norm_vec = Numo::NMath.sqrt((x**2).sum(1))
        self
      end

      # Calculate L2-norms of each sample, and then normalize samples to unit L2-norm.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate L2-norms.
      # @return [Numo::DFloat] The normalized samples.
      def fit_transform(x, _y = nil)
        x = check_convert_sample_array(x)
        fit(x)
        x / @norm_vec.tile(x.shape[1], 1).transpose
      end

      # Calculate L2-norms of each sample, and then normalize samples to unit L2-norm.
      # This method calls the fit_transform method. This method exists for the Pipeline class.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate L2-norms.
      # @return [Numo::DFloat] The normalized samples.
      def transform(x)
        fit_transform(x)
      end
    end
  end
end
