require 'svmkit/base/base_estimator'
require 'svmkit/base/transformer'

module SVMKit
  # This module consists of the classes that perform preprocessings.
  module Preprocessing
    # Normalize samples by scaling each feature to a given range.
    #
    #   normalizer = SVMKit::Preprocessing::MinMaxScaler.new(feature_range: [0.0, 1.0])
    #   new_training_samples = normalizer.fit_transform(training_samples)
    #   new_testing_samples = normalizer.transform(testing_samples)
    class MinMaxScaler
      include Base::BaseEstimator
      include Base::Transformer

      DEFAULT_PARAMS = { # :nodoc:
        feature_range: [0.0, 1.0]
      }.freeze

      # The vector consists of the minimum value for each feature.
      attr_reader :min_vec # :nodoc:

      # The vector consists of the maximum value for each feature.
      attr_reader :max_vec # :nodoc:

      # Creates a new normalizer for scaling each feature to a given range.
      #
      # call-seq:
      #   new(feature_range: [0.0, 1.0]) -> MinMaxScaler
      #
      # * *Arguments* :
      #   - +:feature_range+ (Array) (defaults to: [0.0, 1.0]) -- The desired range of samples.
      def initialize(params = {})
        @params = DEFAULT_PARAMS.merge(Hash[params.map { |k, v| [k.to_sym, v] }])
        @min_vec = nil
        @max_vec = nil
      end

      # Calculate the minimum and maximum value of each feature for scaling.
      #
      # :call-seq:
      #   fit(x) -> MinMaxScaler
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to calculate the minimum and maximum values.
      # * *Returns* :
      #   - MinMaxScaler
      def fit(x, _y = nil)
        @min_vec = x.min(0)
        @max_vec = x.max(0)
        self
      end

      # Calculate the minimum and maximum values, and then normalize samples to feature_range.
      #
      # :call-seq:
      #   fit_transform(x) -> NMatrix
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to calculate the minimum and maximum values.
      # * *Returns* :
      #   - The scaled samples (NMatrix)
      def fit_transform(x, _y = nil)
        fit(x).transform(x)
      end

      # Perform scaling the given samples according to feature_range.
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
        dif_vec = @max_vec - @min_vec
        nx = (x - @min_vec.repeat(n_samples, 0)) / dif_vec.repeat(n_samples, 0)
        nx * (@params[:feature_range][1] - @params[:feature_range][0]) + @params[:feature_range][0]
      end

      # Serializes object through Marshal#dump.
      def marshal_dump # :nodoc:
        { params: @params,
          min_vec: Utils.dump_nmatrix(@min_vec),
          max_vec: Utils.dump_nmatrix(@max_vec) }
      end

      # Deserialize object through Marshal#load.
      def marshal_load(obj) # :nodoc:
        @params = obj[:params]
        @min_vec = Utils.restore_nmatrix(obj[:min_vec])
        @max_vec = Utils.restore_nmatrix(obj[:max_vec])
        nil
      end
    end
  end
end
