# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/evaluation_measure/r2_score'

module SVMKit
  module Base
    # Module for all regressors in SVMKit.
    module Regressor
      include Validation

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
        check_sample_array(x)
        check_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        evaluator = SVMKit::EvaluationMeasure::R2Score.new
        evaluator.score(y, predict(x))
      end
    end
  end
end
