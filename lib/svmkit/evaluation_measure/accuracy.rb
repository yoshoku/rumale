# frozen_string_literal: true

require 'svmkit/base/evaluator'

module SVMKit
  # This module consists of the classes for model evaluation.
  module EvaluationMeasure
    # Accuracy is a class that calculates the accuracy of classifier from the predicted labels.
    #
    # @example
    #   evaluator = SVMKit::EvaluationMeasure::Accuracy.new
    #   puts evaluator.score(ground_truth, predicted)
    class Accuracy
      include Base::Evaluator

      # Calculate mean accuracy.
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::Int32] (shape: [n_samples]) Predicted labels.
      # @return [Float] Mean accuracy
      def score(y_true, y_pred)
        check_label_array(y_true)
        check_label_array(y_pred)

        (y_true.to_a.map.with_index { |label, n| label == y_pred[n] ? 1 : 0 }).inject(:+) / y_true.size.to_f
      end
    end
  end
end
