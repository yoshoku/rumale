# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/evaluation_measure/purity'

module Rumale
  module Base
    # Module for all clustering algorithms in Rumale.
    module ClusterAnalyzer
      include Validation

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
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)
        evaluator = Rumale::EvaluationMeasure::Purity.new
        evaluator.score(y, fit_predict(x))
      end
    end
  end
end
