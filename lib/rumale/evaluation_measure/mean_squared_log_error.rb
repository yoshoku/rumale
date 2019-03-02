# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # MeanSquaredLogError is a class that calculates the mean squared logarithmic error.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::MeanSquaredError.new
    #   puts evaluator.score(ground_truth, predicted)
    class MeanSquaredLogError
      include Base::Evaluator

      # Calculate mean squared logarithmic error.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples, n_outputs]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_outputs]) Estimated target values.
      # @return [Float] Mean squared logarithmic error.
      def score(y_true, y_pred)
        check_tvalue_array(y_true)
        check_tvalue_array(y_pred)
        raise ArgumentError, 'Expect to have the same size both y_true and y_pred.' unless y_true.shape == y_pred.shape

        ((Numo::NMath.log(y_true + 1) - Numo::NMath.log(y_pred + 1))**2).mean
      end
    end
  end
end
