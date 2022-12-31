# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/validation'

module Rumale
  module Preprocessing
    # Normalize samples to unit L1-norm.
    #
    # @example
    #   require 'rumale/preprocessing/l1_normalizer'
    #
    #   normalizer = Rumale::Preprocessing::L1Normalizer.new
    #   new_samples = normalizer.fit_transform(samples)
    class L1Normalizer < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Return the vector consists of L1-norm for each sample.
      # @return [Numo::DFloat] (shape: [n_samples])
      attr_reader :norm_vec # :nodoc:

      # Create a new normalizer for normaliing to L1-norm.
      def initialize # rubocop:disable Lint/UselessMethodDefinition
        super()
      end

      # Calculate L1-norms of each sample.
      #
      # @overload fit(x) -> L1Normalizer
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate L1-norms.
      # @return [L1Normalizer]
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        @norm_vec = x.abs.sum(axis: 1)
        @norm_vec[@norm_vec.eq(0)] = 1
        self
      end

      # Calculate L1-norms of each sample, and then normalize samples to L1-norm.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate L1-norms.
      # @return [Numo::DFloat] The normalized samples.
      def fit_transform(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit(x)
        x / @norm_vec.expand_dims(1)
      end

      # Calculate L1-norms of each sample, and then normalize samples to L1-norm.
      # This method calls the fit_transform method. This method exists for the Pipeline class.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate L1-norms.
      # @return [Numo::DFloat] The normalized samples.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit_transform(x)
      end
    end
  end
end
