# frozen_string_literal: true

require 'numo/narray/alt'

module Rumale
  module Base
    # Module for all regressors in Rumale.
    module Regressor
      # An abstract method for fitting a model.
      def fit
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # An abstract method for predicting labels.
      def predict
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # Calculate the coefficient of determination for the given testing data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) Testing data.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) Target values for testing data.
      # @return [Float] Coefficient of determination
      def score(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        predicted = predict(x)
        n_samples, n_outputs = y.shape
        numerator = ((y - predicted)**2).sum(axis: 0)
        yt_mean = y.sum(axis: 0) / n_samples
        denominator = ((y - yt_mean)**2).sum(axis: 0)
        if n_outputs.nil?
          denominator.zero? ? 0.0 : 1.0 - numerator.fdiv(denominator)
        else
          scores = 1.0 - numerator / denominator
          scores[denominator.eq(0)] = 0.0
          scores.sum.fdiv(scores.size)
        end
      end
    end
  end
end
