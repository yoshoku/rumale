# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Encode labels to binary labels with one-vs-all scheme.
    #
    # @example
    #   encoder = Rumale::Preprocessing::LabelBinarizer.new
    #   label = [0, -1, 3, 3, 1, 1]
    #   p encoder.fit_transform(label)
    #   # Numo::Int32#shape=[6,4]
    #   # [[0, 1, 0, 0],
    #   #  [1, 0, 0, 0],
    #   #  [0, 0, 0, 1],
    #   #  [0, 0, 0, 1],
    #   #  [0, 0, 1, 0],
    #   #  [0, 0, 1, 0]]
    class LabelBinarizer
      include Base::BaseEstimator
      include Base::Transformer

      # Return the class labels.
      # @return [Array] (size: [n_classes])
      attr_reader :classes

      # Create a new encoder for binarizing labels with one-vs-all scheme.
      #
      # @param neg_label [Integer] The value represents negative label.
      # @param pos_label [Integer] The value represents positive label.
      def initialize(neg_label: 0, pos_label: 1)
        check_params_integer(neg_label: neg_label, pos_label: pos_label)
        @params = {}
        @params[:neg_label] = neg_label
        @params[:pos_label] = pos_label
        @classes = nil
      end

      # Fit encoder to labels.
      #
      # @overload fit(y) -> LabelBinarizer
      #   @param y [Numo::NArray/Array] (shape: [n_samples]) The labels to fit encoder.
      # @return [LabelBinarizer]
      def fit(y, _not_used = nil)
        y = y.to_a if y.is_a?(Numo::NArray)
        check_params_type(Array, y: y)
        @classes = y.uniq.sort
        self
      end

      # Fit encoder to labels, then return binarized labels.
      #
      # @overload fit_transform(y) -> Numo::DFloat
      #   @param y [Numo::NArray/Array] (shape: [n_samples]) The labels to fit encoder.
      # @return [Numo::Int32] (shape: [n_samples, n_classes]) The binarized labels.
      def fit_transform(y, _not_used = nil)
        y = y.to_a if y.is_a?(Numo::NArray)
        check_params_type(Array, y: y)
        fit(y).transform(y)
      end

      # Encode labels.
      #
      # @param y [Array] (shape: [n_samples]) The labels to be encoded.
      # @return [Numo::Int32] (shape: [n_samples, n_classes]) The binarized labels.
      def transform(y)
        y = y.to_a if y.is_a?(Numo::NArray)
        check_params_type(Array, y: y)
        n_classes = @classes.size
        n_samples = y.size
        codes = Numo::Int32.zeros(n_samples, n_classes) + @params[:neg_label]
        n_samples.times { |n| codes[n, @classes.index(y[n])] = @params[:pos_label] }
        codes
      end

      # Decode binarized labels.
      #
      # @param x [Numo::Int32] (shape: [n_samples, n_classes]) The binarized labels to be decoded.
      # @return [Array] (shape: [n_samples]) The decoded labels.
      def inverse_transform(x)
        x = Numo::Int32.cast(x) unless x.is_a?(Numo::Int32)
        n_samples = x.shape[0]
        Array.new(n_samples) { |n| @classes[x[n, true].ne(@params[:neg_label]).where[0]] }
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about LabelBinarizer.
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
