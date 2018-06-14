# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/evaluation_measure/purity'

module SVMKit
  module Base
    # Module for all clustering algorithms in SVMKit.
    module ClusterAnalyzer
      # An abstract method for analyzing clusters and predicting cluster indices.
      def fit_predict
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # Calculate purity of clustering result.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) Testing data.
      # @param y [Numo::Int32] (shape: [n_samples]) True labels for testing data.
      # @return [Float] Purity
      def score(x, y)
        SVMKit::Validation.check_sample_array(x)
        SVMKit::Validation.check_label_array(y)
        SVMKit::Validation.check_sample_label_size(x, y)
        evaluator = SVMKit::EvaluationMeasure::Purity.new
        evaluator.score(y, fit_predict(x))
      end
    end
  end
end
