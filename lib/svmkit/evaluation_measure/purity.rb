# frozen_string_literal: true

require 'svmkit/base/evaluator'

module SVMKit
  module EvaluationMeasure
    # Purity is a class that calculates the purity of cluatering results.
    #
    # @example
    #   evaluator = SVMKit::EvaluationMeasure::Purity.new
    #   puts evaluator.score(ground_truth, predicted)
    #
    # *Reference*
    # - C D. Manning, P. Raghavan, and H. Schutze, "Introduction to Information Retrieval," Cambridge University Press., 2008.
    class Purity
      include Base::Evaluator

      # Calculate purity
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth labels.
      # @param y_pred [Numo::Int32] (shape: [n_samples]) Predicted cluster labels.
      # @return [Float] Purity
      def score(y_true, y_pred)
        check_label_array(y_true)
        check_label_array(y_pred)
        # initiazlie some variables.
        purity = 0
        n_samples = y_pred.size
        class_ids = y_true.to_a.uniq
        cluster_ids = y_pred.to_a.uniq
        # calculate purity.
        cluster_ids.each do |k|
          pr_sample_ids = y_pred.eq(k).where.to_a
          purity += class_ids.map { |j| (pr_sample_ids & y_true.eq(j).where.to_a).size }.max
        end
        purity.fdiv(n_samples)
      end
    end
  end
end
