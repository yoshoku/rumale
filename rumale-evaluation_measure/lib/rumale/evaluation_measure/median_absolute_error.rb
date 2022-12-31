# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # MedianAbsoluteError is a class that calculates the median absolute error.
    #
    # @example
    #   require 'rumale/evaluation_measure/median_absolute_error'
    #
    #   evaluator = Rumale::EvaluationMeasure::MedianAbsoluteError.new
    #   puts evaluator.score(ground_truth, predicted)
    class MedianAbsoluteError
      include ::Rumale::Base::Evaluator

      # Calculate median absolute error.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples]) Estimated target values.
      # @return [Float] Median absolute error.
      def score(y_true, y_pred)
        (y_true - y_pred).abs.median
      end
    end
  end
end
