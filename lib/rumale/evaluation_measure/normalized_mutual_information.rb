# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # NormalizedMutualInformation is a class that calculates the normalized mutual information of cluatering results.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::NormalizedMutualInformation.new
    #   puts evaluator.score(ground_truth, predicted)
    #
    # *Reference*
    # - C D. Manning, P. Raghavan, and H. Schutze, "Introduction to Information Retrieval," Cambridge University Press., 2008.
    # - N X. Vinh, J. Epps, and J. Bailey, "Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance," J. Machine Learning Research, vol. 11, pp. 2837--1854, 2010.
    class NormalizedMutualInformation
      include Base::Evaluator

      # Calculate noramlzied mutual information
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::Int32] (shape: [n_samples]) Predicted cluster labels.
      # @return [Float] Normalized mutual information
      def score(y_true, y_pred)
        check_label_array(y_true)
        check_label_array(y_pred)
        # initiazlie some variables.
        mutual_information = 0.0
        n_samples = y_pred.size
        class_ids = y_true.to_a.uniq
        cluster_ids = y_pred.to_a.uniq
        # calculate entropy.
        class_entropy = -1.0 * class_ids.map do |k|
          ratio = y_true.eq(k).count.fdiv(n_samples)
          ratio * Math.log(ratio)
        end.reduce(:+)
        return 0.0 if class_entropy.zero?
        cluster_entropy = -1.0 * cluster_ids.map do |k|
          ratio = y_pred.eq(k).count.fdiv(n_samples)
          ratio * Math.log(ratio)
        end.reduce(:+)
        return 0.0 if cluster_entropy.zero?
        # calculate mutual information.
        cluster_ids.map do |k|
          pr_sample_ids = y_pred.eq(k).where.to_a
          n_pr_samples = pr_sample_ids.size
          class_ids.map do |j|
            tr_sample_ids = y_true.eq(j).where.to_a
            n_tr_samples = tr_sample_ids.size
            n_intr_samples = (pr_sample_ids & tr_sample_ids).size
            if n_intr_samples.positive?
              mutual_information +=
                n_intr_samples.fdiv(n_samples) * Math.log((n_samples * n_intr_samples).fdiv(n_pr_samples * n_tr_samples))
            end
          end
        end
        # return normalized mutual information.
        mutual_information / Math.sqrt(class_entropy * cluster_entropy)
      end
    end
  end
end
