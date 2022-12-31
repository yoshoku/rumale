# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # LogLoss is a class that calculates the logarithmic loss of predicted class probability.
    #
    # @example
    #   require 'rumale/evaluation_measure/log_loss'
    #
    #   evaluator = Rumale::EvaluationMeasure::LogLoss.new
    #   puts evaluator.score(ground_truth, predicted)
    class LogLoss
      include ::Rumale::Base::Evaluator

      # Calculate mean logarithmic loss.
      # If both y_true and y_pred are array (both shapes are [n_samples]), this method calculates
      # mean logarithmic loss for binary classification.
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted class probability.
      # @param eps [Float] A small value close to zero to avoid outputting infinity in logarithmic calcuation.
      # @return [Float] mean logarithmic loss
      def score(y_true, y_pred, eps = 1e-15)
        n_samples, n_classes = y_pred.shape
        clipped_p = y_pred.clip(eps, 1 - eps)

        log_loss = if n_classes.nil?
                     negative_label = y_true.to_a.uniq.min
                     bin_y_true = Numo::DFloat.cast(y_true.ne(negative_label))
                     -(bin_y_true * Numo::NMath.log(clipped_p) + (1 - bin_y_true) * Numo::NMath.log(1 - clipped_p))
                   else
                     binarized_y_true = binarize(y_true)
                     clipped_p /= clipped_p.sum(axis: 1).expand_dims(1)
                     -(binarized_y_true * Numo::NMath.log(clipped_p)).sum(axis: 1)
                   end
        log_loss.sum / n_samples
      end

      private

      def binarize(y)
        classes = y.to_a.uniq.sort
        n_samples = y.size
        n_classes = classes.size
        binarized = Numo::DFloat.zeros(n_samples, n_classes)
        n_samples.times { |n| binarized[n, classes.index(y[n])] = 1 }
        binarized
      end
    end
  end
end
