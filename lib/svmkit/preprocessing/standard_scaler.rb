require 'svmkit/base/base_estimator'
require 'svmkit/base/transformer'

module SVMKit
  # This module consists of the classes that perform preprocessings.
  module Preprocessing
    # Normalize samples by centering and scaling to unit variance.
    #
    #   normalizer = SVMKit::Preprocessing::StandardScaler.new
    #   new_training_samples = normalizer.fit_transform(training_samples)
    #   new_testing_samples = normalizer.transform(testing_samples)
    class StandardScaler
      include Base::BaseEstimator
      include Base::Transformer

      # The vector consists of the mean value for each feature.
      attr_reader :mean_vec # :nodoc:

      # The vector consists of the standard deviation for each feature.
      attr_reader :std_vec # :nodoc:

      # Create a new normalizer for centering and scaling to unit variance.
      #
      # :call-seq:
      #   new() -> StandardScaler
      def initialize(_params = {})
        @mean_vec = nil
        @std_vec = nil
      end

      # Calculate the mean value and standard deviation of each feature for scaling.
      #
      # :call-seq:
      #   fit(x) -> StandardScaler
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to calculate the mean values and standard deviations.
      # * *Returns* :
      #   - StandardScaler
      def fit(x, _y = nil)
        @mean_vec = x.mean(0)
        @std_vec = x.std(0)
        self
      end

      # Calculate the mean values and standard deviations, and then normalize samples using them.
      #
      # :call-seq:
      #   fit_transform(x) -> NMatrix
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to calculate the mean values and standard deviations.
      # * *Returns* :
      #   - The scaled samples (NMatrix)
      def fit_transform(x, _y = nil)
        fit(x).transform(x)
      end

      # Perform standardization the given samples.
      #
      # call-seq:
      #   transform(x) -> NMatrix
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to be scaled.
      # * *Returns* :
      #   - The scaled samples (NMatrix)
      def transform(x)
        n_samples, = x.shape
        (x - @mean_vec.repeat(n_samples, 0)) / @std_vec.repeat(n_samples, 0)
      end

      # Serializes object through Marshal#dump.
      def marshal_dump # :nodoc:
        { mean_vec: Utils.dump_nmatrix(@mean_vec),
          std_vec: Utils.dump_nmatrix(@std_vec) }
      end

      # Deserialize object through Marshal#load.
      def marshal_load(obj) # :nodoc:
        @mean_vec = Utils.restore_nmatrix(obj[:mean_vec])
        @std_vec = Utils.restore_nmatrix(obj[:std_vec])
        nil
      end
    end
  end
end
