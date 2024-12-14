# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Encode categorical integer features to one-hot-vectors.
    #
    # @example
    #   require 'rumale/preprocessing/one_hot_encoder'
    #
    #   encoder = Rumale::Preprocessing::OneHotEncoder.new
    #   labels = Numo::Int32[0, 0, 2, 3, 2, 1]
    #   one_hot_vectors = encoder.fit_transform(labels)
    #   # > pp one_hot_vectors
    #   # Numo::DFloat#shape[6, 4]
    #   # [[1, 0, 0, 0],
    #   #  [1, 0, 0, 0],
    #   #  [0, 0, 1, 0],
    #   #  [0, 0, 0, 1],
    #   #  [0, 0, 1, 0],
    #   #  [0, 1, 0, 0]]
    class OneHotEncoder < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Return the maximum values for each feature.
      # @return [Numo::Int32] (shape: [n_features])
      attr_reader :n_values

      # Return the indices for feature values that actually occur in the training set.
      # @return [Nimo::Int32]
      attr_reader :active_features

      # Return the indices to feature ranges.
      # @return [Numo::Int32] (shape: [n_features + 1])
      attr_reader :feature_indices

      # Create a new encoder for encoding categorical integer features to one-hot-vectors
      def initialize # rubocop:disable Lint/UselessMethodDefinition
        super
      end

      # Fit one-hot-encoder to samples.
      #
      # @overload fit(x) -> OneHotEncoder
      #   @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to fit one-hot-encoder.
      # @return [OneHotEncoder]
      def fit(x, _y = nil)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?

        @n_values = x.max(0) + 1
        @feature_indices = Numo::Int32.hstack([[0], @n_values]).cumsum
        @active_features = encode(x, @feature_indices).sum(axis: 0).ne(0).where
        self
      end

      # Fit one-hot-encoder to samples, then encode samples into one-hot-vectors
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to encode into one-hot-vectors.
      # @return [Numo::DFloat] The one-hot-vectors.
      def fit_transform(x, _y = nil)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?

        fit(x).transform(x)
      end

      # Encode samples into one-hot-vectors.
      #
      # @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to encode into one-hot-vectors.
      # @return [Numo::DFloat] The one-hot-vectors.
      def transform(x)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?

        codes = encode(x, @feature_indices)
        codes[true, @active_features].dup
      end

      private

      def encode(x, indices)
        n_samples, n_features = x.shape
        n_features = 1 if n_features.nil?
        col_indices = (x + indices[0...-1]).flatten.to_a
        row_indices = Numo::Int32.new(n_samples).seq.repeat(n_features).to_a
        codes = Numo::DFloat.zeros(n_samples, indices[-1])
        row_indices.zip(col_indices).each { |r, c| codes[r, c] = 1.0 }
        codes
      end
    end
  end
end
