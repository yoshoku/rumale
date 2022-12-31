# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # R2Score is a class that calculates the coefficient of determination for the predicted values.
    #
    # @example
    #   require 'rumale/evaluation_measure/r2_score'
    #
    #   evaluator = Rumale::EvaluationMeasure::R2Score.new
    #   puts evaluator.score(ground_truth, predicted)
    class R2Score
      include ::Rumale::Base::Evaluator

      # Create a new evaluation measure calculater for coefficient of determination.
      def initialize; end

      # Calculate the coefficient of determination.
      #
      # @param y_true [Numo::DFloat] (shape: [n_samples, n_outputs]) Ground truth target values.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_outputs]) Estimated taget values.
      # @return [Float] Coefficient of determination
      def score(y_true, y_pred)
        n_samples, n_outputs = y_true.shape
        numerator = ((y_true - y_pred)**2).sum(axis: 0)
        yt_mean = y_true.sum(axis: 0) / n_samples
        denominator = ((y_true - yt_mean)**2).sum(axis: 0)
        if n_outputs.nil?
          denominator.zero? ? 0.0 : 1.0 - numerator / denominator
        else
          scores = 1 - numerator / denominator
          scores[denominator.eq(0)] = 0.0
          scores.sum / scores.size
        end
      end
    end
  end
end
