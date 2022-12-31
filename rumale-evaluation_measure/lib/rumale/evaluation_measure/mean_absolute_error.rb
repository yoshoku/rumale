# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # MeanAbsoluteError is a class that calculates the mean absolute error.
    #
    # @example
    #   require 'rumale/evaluation_measure/mean_absolute_error'
    #
    #   evaluator = Rumale::EvaluationMeasure::MeanAbsoluteError.new
    #   puts evaluator.score(ground_truth, predicted)
    class MeanAbsoluteError
      include ::Rumale::Base::Evaluator

      # Calculate mean absolute error.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples, n_outputs]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_outputs]) Estimated target values.
      # @return [Float] Mean absolute error
      def score(y_true, y_pred)
        (y_true - y_pred).abs.mean
      end
    end
  end
end
