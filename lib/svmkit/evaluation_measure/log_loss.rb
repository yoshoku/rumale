# frozen_string_literal: true

require 'svmkit/base/evaluator'

module SVMKit
  module EvaluationMeasure
    # LogLoss is a class that calculates the logarithmic loss of predicted class probability.
    #
    # @example
    #   evaluator = SVMKit::EvaluationMeasure::LogLoss.new
    #   puts evaluator.score(ground_truth, predicted)
    class LogLoss
      include Base::Evaluator

      # Claculate mean logarithmic loss.
      # If both y_true and y_pred are array (both shapes are [n_samples]), this method calculates
      # mean logarithmic loss for binary classification.
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted class probability.
      # @param eps [Float] A small value close to zero to avoid outputting infinity in logarithmic calcuation.
      # @return [Float] mean logarithmic loss
      def score(y_true, y_pred, eps = 1e-15)
        SVMKit::Validation.check_params_type(Numo::Int32, y_true: y_true)
        SVMKit::Validation.check_params_type(Numo::DFloat, y_pred: y_pred)

        n_samples, n_classes = y_pred.shape
        clipped_p = y_pred.clip(eps, 1 - eps)

        log_loss = if n_classes.nil?
                     negative_label = y_true.to_a.uniq.sort.first
                     bin_y_true = Numo::DFloat.cast(y_true.ne(negative_label))
                     -(bin_y_true * Numo::NMath.log(clipped_p) + (1 - bin_y_true) * Numo::NMath.log(1 - clipped_p))
                   else
                     encoder = SVMKit::Preprocessing::OneHotEncoder.new
                     encoded_y_true = encoder.fit_transform(y_true)
                     clipped_p /= clipped_p.sum(1).expand_dims(1)
                     -(encoded_y_true * Numo::NMath.log(clipped_p)).sum(1)
                   end
        log_loss.sum / n_samples
      end
    end
  end
end
