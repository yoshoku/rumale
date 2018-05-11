# frozen_string_literal: true

require 'svmkit/base/evaluator'

module SVMKit
  module EvaluationMeasure
    # MeanSquaredError is a class that calculates the mean squared error.
    #
    # @example
    #   evaluator = SVMKit::EvaluationMeasure::MeanSquaredError.new
    #   puts evaluator.score(ground_truth, predicted)
    class MeanSquaredError
      include Base::Evaluator

      # Calculate mean squared error.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples, n_outputs]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_outputs]) Estimated target values.
      # @return [Float] Mean squared error
      def score(y_true, y_pred)
        SVMKit::Validation.check_tvalue_array(y_true)
        SVMKit::Validation.check_tvalue_array(y_pred)
        raise ArgumentError, 'Expect to have the same size both y_true and y_pred.' unless y_true.shape == y_pred.shape

        ((y_true - y_pred)**2).mean
      end
    end
  end
end
