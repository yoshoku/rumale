# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Encode categorical integer features to one-hot-vectors.
    #
    # @example
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
    class OneHotEncoder
      include Base::BaseEstimator
      include Base::Transformer

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
      def initialize
        @params = {}
        @n_values = nil
        @active_features = nil
        @feature_indices = nil
      end

      # Fit one-hot-encoder to samples.
      #
      # @overload fit(x) -> OneHotEncoder
      #   @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to fit one-hot-encoder.
      # @return [OneHotEncoder]
      def fit(x, _y = nil)
        x = Numo::Int32.cast(x) unless x.is_a?(Numo::Int32)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?
        @n_values = x.max(0) + 1
        @feature_indices = Numo::Int32.hstack([[0], @n_values]).cumsum
        @active_features = encode(x, @feature_indices).sum(0).ne(0).where
        self
      end

      # Fit one-hot-encoder to samples, then encode samples into one-hot-vectors
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to encode into one-hot-vectors.
      # @return [Numo::DFloat] The one-hot-vectors.
      def fit_transform(x, _y = nil)
        x = Numo::Int32.cast(x) unless x.is_a?(Numo::Int32)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?
        fit(x).transform(x)
      end

      # Encode samples into one-hot-vectors.
      #
      # @param x [Numo::Int32] (shape: [n_samples, n_features]) The samples to encode into one-hot-vectors.
      # @return [Numo::DFloat] The one-hot-vectors.
      def transform(x)
        x = Numo::Int32.cast(x) unless x.is_a?(Numo::Int32)
        raise ArgumentError, 'Expected the input samples only consists of non-negative integer values.' if x.lt(0).any?
        codes = encode(x, @feature_indices)
        codes[true, @active_features].dup
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about OneHotEncoder.
      def marshal_dump
        { params: @params,
          n_values: @n_values,
          active_features: @active_features,
          feature_indices: @feature_indices }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @n_values = obj[:n_values]
        @active_features = obj[:active_features]
        @feature_indices = obj[:feature_indices]
        nil
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
