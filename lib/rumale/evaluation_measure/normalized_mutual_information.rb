# frozen_string_literal: true

require 'rumale/base/evaluator'
require 'rumale/evaluation_measure/mutual_information'

module Rumale
  module EvaluationMeasure
    # NormalizedMutualInformation is a class that calculates the normalized mutual information.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::NormalizedMutualInformation.new
    #   puts evaluator.score(ground_truth, predicted)
    #
    # *Reference*
    # - Manning, C D., Raghavan, P., and Schutze, H., "Introduction to Information Retrieval," Cambridge University Press., 2008.
    # - Vinh, N X., Epps, J., and Bailey, J., "Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance," J. Machine Learning Research, vol. 11, pp. 2837--1854, 2010.
    class NormalizedMutualInformation
      include Base::Evaluator

      # Calculate noramlzied mutual information
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::Int32] (shape: [n_samples]) Predicted cluster labels.
      # @return [Float] Normalized mutual information
      def score(y_true, y_pred)
        y_true = check_convert_label_array(y_true)
        y_pred = check_convert_label_array(y_pred)
        # calculate entropies.
        class_entropy = entropy(y_true)
        return 0.0 if class_entropy.zero?

        cluster_entropy = entropy(y_pred)
        return 0.0 if cluster_entropy.zero?

        # calculate mutual information.
        mi = MutualInformation.new
        mi.score(y_true, y_pred) / Math.sqrt(class_entropy * cluster_entropy)
      end

      private

      def entropy(y)
        n_samples = y.size
        indices = y.to_a.uniq
        sum_log = indices.map do |k|
          ratio = y.eq(k).count.fdiv(n_samples)
          ratio * Math.log(ratio)
        end.reduce(:+)
        -sum_log
      end
    end
  end
end
