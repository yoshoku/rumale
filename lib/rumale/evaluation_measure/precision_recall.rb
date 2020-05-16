# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  # This module consists of the classes for model evaluation.
  module EvaluationMeasure
    # @!visibility private
    module PrecisionRecall
      module_function

      # @!visibility private
      def precision_each_class(y_true, y_pred)
        y_true.sort.to_a.uniq.map do |label|
          target_positions = y_pred.eq(label)
          next 0.0 if y_pred[target_positions].empty?

          n_true_positives = Numo::Int32.cast(y_true[target_positions].eq(y_pred[target_positions])).sum.to_f
          n_false_positives = Numo::Int32.cast(y_true[target_positions].ne(y_pred[target_positions])).sum.to_f
          n_true_positives / (n_true_positives + n_false_positives)
        end
      end

      # @!visibility private
      def recall_each_class(y_true, y_pred)
        y_true.sort.to_a.uniq.map do |label|
          target_positions = y_true.eq(label)
          next 0.0 if y_pred[target_positions].empty?

          n_true_positives = Numo::Int32.cast(y_true[target_positions].eq(y_pred[target_positions])).sum.to_f
          n_false_negatives = Numo::Int32.cast(y_true[target_positions].ne(y_pred[target_positions])).sum.to_f
          n_true_positives / (n_true_positives + n_false_negatives)
        end
      end

      # @!visibility private
      def f_score_each_class(y_true, y_pred)
        precision_each_class(y_true, y_pred).zip(recall_each_class(y_true, y_pred)).map do |p, r|
          next 0.0 if p.zero? && r.zero?

          (2.0 * p * r) / (p + r)
        end
      end

      # @!visibility private
      def micro_average_precision(y_true, y_pred)
        evaluated_values = y_true.sort.to_a.uniq.map do |label|
          target_positions = y_pred.eq(label)
          next [0.0, 0.0] if y_pred[target_positions].empty?

          n_true_positives = Numo::Int32.cast(y_true[target_positions].eq(y_pred[target_positions])).sum.to_f
          n_false_positives = Numo::Int32.cast(y_true[target_positions].ne(y_pred[target_positions])).sum.to_f
          [n_true_positives, n_true_positives + n_false_positives]
        end
        res = evaluated_values.transpose.map { |v| v.inject(:+) }
        res.first / res.last
      end

      # @!visibility private
      def micro_average_recall(y_true, y_pred)
        evaluated_values = y_true.sort.to_a.uniq.map do |label|
          target_positions = y_true.eq(label)
          next 0.0 if y_pred[target_positions].empty?

          n_true_positives = Numo::Int32.cast(y_true[target_positions].eq(y_pred[target_positions])).sum.to_f
          n_false_negatives = Numo::Int32.cast(y_true[target_positions].ne(y_pred[target_positions])).sum.to_f
          [n_true_positives, n_true_positives + n_false_negatives]
        end
        res = evaluated_values.transpose.map { |v| v.inject(:+) }
        res.first / res.last
      end

      # @!visibility private
      def micro_average_f_score(y_true, y_pred)
        p = micro_average_precision(y_true, y_pred)
        r = micro_average_recall(y_true, y_pred)
        (2.0 * p * r) / (p + r)
      end

      # @!visibility private
      def macro_average_precision(y_true, y_pred)
        precision_each_class(y_true, y_pred).inject(:+) / y_true.to_a.uniq.size
      end

      # @!visibility private
      def macro_average_recall(y_true, y_pred)
        recall_each_class(y_true, y_pred).inject(:+) / y_true.to_a.uniq.size
      end

      # @!visibility private
      def macro_average_f_score(y_true, y_pred)
        f_score_each_class(y_true, y_pred).inject(:+) / y_true.to_a.uniq.size
      end
    end
  end
end
