# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # MedianAbsoluteError is a class that calculates the median absolute error.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::MedianAbsoluteError.new
    #   puts evaluator.score(ground_truth, predicted)
    class MedianAbsoluteError
      include Base::Evaluator

      # Calculate median absolute error.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples]) Estimated target values.
      # @return [Float] Median absolute error.
      def score(y_true, y_pred)
        check_tvalue_array(y_true)
        check_tvalue_array(y_pred)
        raise ArgumentError, 'Expect to have the same size both y_true and y_pred.' unless y_true.shape == y_pred.shape
        raise ArgumentError, 'Expect target values to be 1-D arrray' if [y_true.shape.size, y_pred.shape.size].max > 1

        (y_true - y_pred).abs.median
      end
    end
  end
end
