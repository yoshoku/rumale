# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # AdjustedRandScore is a class that calculates the adjusted rand index.
    #
    # @example
    #   require 'rumale/evaluation_measure/adjusted_rand_score'
    #
    #   evaluator = Rumale::EvaluationMeasure::AdjustedRandScore.new
    #   puts evaluator.score(ground_truth, predicted)
    #
    # *Reference*
    # - Vinh, N X., Epps, J., and Bailey, J., "Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance", J. Machine Learnig Research, Vol. 11, pp.2837--2854, 2010.
    class AdjustedRandScore
      include ::Rumale::Base::Evaluator

      # Calculate adjusted rand index.
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::Int32] (shape: [n_samples]) Predicted cluster labels.
      # @return [Float] Adjusted rand index.
      def score(y_true, y_pred)
        # initiazlie some variables.
        n_samples = y_pred.size
        n_classes = y_true.to_a.uniq.size
        n_clusters = y_pred.to_a.uniq.size

        # check special cases.
        return 1.0 if special_cases?(n_samples, n_classes, n_clusters)

        # calculate adjusted rand index.
        table = contingency_table(y_true, y_pred)
        sum_comb_a = table.sum(axis: 1).to_a.sum { |v| comb_two(v) }
        sum_comb_b = table.sum(axis: 0).to_a.sum { |v| comb_two(v) }
        sum_comb = table.flatten.to_a.sum { |v| comb_two(v) }
        prod_comb = (sum_comb_a * sum_comb_b).fdiv(comb_two(n_samples))
        mean_comb = (sum_comb_a + sum_comb_b).fdiv(2)
        (sum_comb - prod_comb).fdiv(mean_comb - prod_comb)
      end

      private

      def contingency_table(y_true, y_pred)
        class_ids = y_true.to_a.uniq
        cluster_ids = y_pred.to_a.uniq
        n_classes = class_ids.size
        n_clusters = cluster_ids.size
        table = Numo::Int32.zeros(n_classes, n_clusters)
        n_classes.times do |i|
          b_true = y_true.eq(class_ids[i])
          n_clusters.times do |j|
            b_pred = y_pred.eq(cluster_ids[j])
            table[i, j] = (b_true & b_pred).count
          end
        end
        table
      end

      def special_cases?(n_samples, n_classes, n_clusters)
        (n_classes.zero? && n_clusters.zero?) ||
          (n_classes == 1 && n_clusters == 1) ||
          (n_classes == n_samples && n_clusters == n_samples)
      end

      def comb_two(k)
        k * (k - 1) / 2
      end
    end
  end
end
