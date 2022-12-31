# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # MeanSquaredError is a class that calculates the mean squared error.
    #
    # @example
    #   require 'rumale/evaluation_measure/mean_squared_error'
    #
    #   evaluator = Rumale::EvaluationMeasure::MeanSquaredError.new
    #   puts evaluator.score(ground_truth, predicted)
    class MeanSquaredError
      include ::Rumale::Base::Evaluator

      # Calculate mean squared error.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples, n_outputs]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_outputs]) Estimated target values.
      # @return [Float] Mean squared error
      def score(y_true, y_pred)
        ((y_true - y_pred)**2).mean
      end
    end
  end
end
