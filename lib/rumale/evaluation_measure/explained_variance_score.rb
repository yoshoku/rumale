# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # ExplainedVarianceScore is a class that calculates the explained variance score.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::ExplainedVarianceScore.new
    #   puts evaluator.score(ground_truth, predicted)
    class ExplainedVarianceScore
      include Base::Evaluator

      # Calculate explained variance score.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples, n_outputs]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_outputs]) Estimated target values.
      # @return [Float] Explained variance score.
      def score(y_true, y_pred)
        y_true = check_convert_tvalue_array(y_true)
        y_pred = check_convert_tvalue_array(y_pred)
        raise ArgumentError, 'Expect to have the same size both y_true and y_pred.' unless y_true.shape == y_pred.shape

        diff = y_true - y_pred
        numerator = ((diff - diff.mean(0))**2).mean(0)
        denominator = ((y_true - y_true.mean(0))**2).mean(0)

        n_outputs = y_true.shape[1]
        if n_outputs.nil?
          denominator.zero? ? 0 : 1.0 - numerator / denominator
        else
          valids = denominator.ne(0)
          (1.0 - numerator[valids] / denominator[valids]).sum / n_outputs
        end
      end
    end
  end
end
