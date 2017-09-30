require 'svmkit/base/base_estimator'
require 'svmkit/base/transformer'

module SVMKit
  # This module consists of the classes that perform preprocessings.
  module Preprocessing
    # Normalize samples to unit L2-norm.
    #
    #   normalizer = SVMKit::Preprocessing::StandardScaler.new
    #   new_samples = normalizer.fit_transform(samples)
    class L2Normalizer
      include Base::BaseEstimator
      include Base::Transformer

      # The vector consists of norms of each sample.
      attr_reader :norm_vec # :nodoc:

      # Create a new normalizer for normaliing to unit L2-norm.
      #
      # :call-seq:
      #   new() -> L2Normalizer
      def initialize(_params = {})
        @norm_vec = nil
      end

      # Calculate L2 norms of each sample.
      #
      # :call-seq:
      #   fit(x) -> L2Normalizer
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to calculate L2-norms.
      # * *Returns* :
      #   - L2Normalizer
      def fit(x, _y = nil)
        n_samples, = x.shape
        @norm_vec = NMatrix.new([1, n_samples],
                                Array.new(n_samples) { |n| x.row(n).norm2 })
        self
      end

      # Calculate L2 norms of each sample, and then normalize samples to unit L2-norm.
      #
      # :call-seq:
      #   fit_transform(x) -> NMatrix
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to calculate L2-norms.
      # * *Returns* :
      #   - The normalized samples (NMatrix)
      def fit_transform(x, _y = nil)
        fit(x)
        x / @norm_vec.transpose.repeat(x.shape[1], 1)
      end
    end
  end
end
