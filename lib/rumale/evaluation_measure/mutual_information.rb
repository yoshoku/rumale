# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # MutualInformation is a class that calculates the mutual information.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::MutualInformation.new
    #   puts evaluator.score(ground_truth, predicted)
    #
    # *Reference*
    # - Vinh, N X., Epps, J., and Bailey, J., "Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance," J. Machine Learning Research, vol. 11, pp. 2837--1854, 2010.
    class MutualInformation
      include Base::Evaluator

      # Calculate mutual information
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::Int32] (shape: [n_samples]) Predicted cluster labels.
      # @return [Float] Mutual information.
      def score(y_true, y_pred)
        y_true = check_convert_label_array(y_true)
        y_pred = check_convert_label_array(y_pred)
        # initiazlie some variables.
        mutual_information = 0.0
        n_samples = y_pred.size
        class_ids = y_true.to_a.uniq
        cluster_ids = y_pred.to_a.uniq
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
        mutual_information
      end
    end
  end
end
