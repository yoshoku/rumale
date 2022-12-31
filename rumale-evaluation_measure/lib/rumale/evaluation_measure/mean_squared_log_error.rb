# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # MeanSquaredLogError is a class that calculates the mean squared logarithmic error.
    #
    # @example
    #   require 'rumale/evaluation_measure/mean_squared_log_error'
    #
    #   evaluator = Rumale::EvaluationMeasure::MeanSquaredLogError.new
    #   puts evaluator.score(ground_truth, predicted)
    class MeanSquaredLogError
      include ::Rumale::Base::Evaluator

      # Calculate mean squared logarithmic error.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples, n_outputs]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_outputs]) Estimated target values.
      # @return [Float] Mean squared logarithmic error.
      def score(y_true, y_pred)
        ((Numo::NMath.log(y_true + 1) - Numo::NMath.log(y_pred + 1))**2).mean
      end
    end
  end
end
