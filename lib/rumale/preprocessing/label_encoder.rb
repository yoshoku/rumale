# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Encode labels to values between 0 and n_classes - 1.
    #
    # @example
    #   encoder = Rumale::Preprocessing::LabelEncoder.new
    #   labels = Numo::Int32[1, 8, 8, 15, 0]
    #   encoded_labels = encoder.fit_transform(labels)
    #   # > pp encoded_labels
    #   # Numo::Int32#shape=[5]
    #   # [1, 2, 2, 3, 0]
    #   decoded_labels = encoder.inverse_transform(encoded_labels)
    #   # > pp decoded_labels
    #   # [1, 8, 8, 15, 0]
    class LabelEncoder
      include Base::BaseEstimator
      include Base::Transformer

      # Return the class labels.
      # @return [Array] (size: [n_classes])
      attr_reader :classes

      # Create a new encoder for encoding labels to values between 0 and n_classes - 1.
      def initialize
        @params = {}
        @classes = nil
      end

      # Fit label-encoder to labels.
      #
      # @overload fit(x) -> LabelEncoder
      #
      # @param x [Array] (shape: [n_samples]) The labels to fit label-encoder.
      # @return [LabelEncoder]
      def fit(x, _y = nil)
        x = x.to_a if x.is_a?(Numo::NArray)
        check_params_type(Array, x: x)
        @classes = x.sort.uniq
        self
      end

      # Fit label-encoder to labels, then return encoded labels.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Array] (shape: [n_samples]) The labels to fit label-encoder.
      # @return [Numo::Int32] The encoded labels.
      def fit_transform(x, _y = nil)
        x = x.to_a if x.is_a?(Numo::NArray)
        check_params_type(Array, x: x)
        fit(x).transform(x)
      end

      # Encode labels.
      #
      # @param x [Array] (shape: [n_samples]) The labels to be encoded.
      # @return [Numo::Int32] The encoded labels.
      def transform(x)
        x = x.to_a if x.is_a?(Numo::NArray)
        check_params_type(Array, x: x)
        Numo::Int32[*(x.map { |v| @classes.index(v) })]
      end

      # Decode encoded labels.
      #
      # @param x [Numo::Int32] (shape: [n_samples]) The labels to be decoded.
      # @return [Array] The decoded labels.
      def inverse_transform(x)
        x = check_convert_label_array(x)
        x.to_a.map { |n| @classes[n] }
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about LabelEncoder
      def marshal_dump
        { params: @params,
          classes: @classes }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @classes = obj[:classes]
        nil
      end
    end
  end
end
