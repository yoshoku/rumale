# frozen_string_literal: true

module SVMKit
  module Base
    # Module for all classifiers in SVMKit.
    module Classifier
      # An abstract method for fitting a model.
      def fit
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # An abstract method for predicting labels.
      def predict
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # Claculate the mean accuracy of the given testing data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) Testing data.
      # @param y [Numo::Int32] (shape: [n_samples]) True labels for testing data.
      # @return [Float] Mean accuracy
      def score(x, y)
        evaluator = SVMKit::EvaluationMeasure::Accuracy.new
        evaluator.score(y, predict(x))
      end
    end
  end
end
