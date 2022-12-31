# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/validation'

module Rumale
  module Preprocessing
    # Binarize samples according to a threshold
    #
    # @example
    #   require 'rumale/preprocessing/binarizer'
    #
    #   binarizer = Rumale::Preprocessing::Binarizer.new
    #   x = Numo::DFloat[[-1.2, 3.2], [2.4, -0.5], [4.5, 0.8]]
    #   b = binarizer.transform(x)
    #   p b
    #
    #   # Numo::DFloat#shape=[3, 2]
    #   # [[0, 1],
    #   #  [1, 0],
    #   #  [1, 1]]
    class Binarizer < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Create a new transformer for binarization.
      # @param threshold [Float] The threshold value for binarization.
      def initialize(threshold: 0.0)
        super()
        @params = { threshold: threshold }
      end

      # This method does nothing and returns the object itself.
      # For compatibility with other transformer, this method exists.
      #
      # @overload fit() -> Binarizer
      #
      # @return [Binarizer]
      def fit(_x = nil, _y = nil)
        self
      end

      # Binarize each sample.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be binarized.
      # @return [Numo::DFloat] The binarized samples.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        x.class.cast(x.gt(@params[:threshold]))
      end

      # The output of this method is the same as that of the transform method.
      # For compatibility with other transformer, this method exists.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be binarized.
      # @return [Numo::DFloat] The binarized samples.
      def fit_transform(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit(x).transform(x)
      end
    end
  end
end
