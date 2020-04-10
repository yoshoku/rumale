# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/evaluation_measure/accuracy'
require 'rumale/evaluation_measure/precision_recall'

module Rumale
  module EvaluationMeasure
    module_function

    # Calculate confusion matrix for evaluating classification performance.
    #
    # @example
    #   y_true = Numo::Int32[2, 0, 2, 2, 0, 1]
    #   y_pred = Numo::Int32[0, 0, 2, 2, 0, 2]
    #   p Rumale::EvaluationMeasure.confusion_matrix(y_true, y_pred)
    #
    #   # Numo::Int32#shape=[3,3]
    #   # [[2, 0, 0],
    #   #  [0, 0, 1],
    #   #  [1, 0, 2]]
    #
    # @param y_true [Numo::Int32] (shape: [n_samples]) The ground truth labels.
    # @param y_pred [Numo::Int32] (shape: [n_samples]) The predicted labels.
    # @return [Numo::Int32] (shape: [n_classes, n_classes]) The confusion matrix.
    def confusion_matrix(y_true, y_pred)
      y_true = Rumale::Validation.check_convert_label_array(y_true)
      y_pred = Rumale::Validation.check_convert_label_array(y_pred)

      labels = y_true.to_a.uniq.sort
      n_labels = labels.size

      conf_mat = Numo::Int32.zeros(n_labels, n_labels)

      labels.each_with_index do |lbl_a, i|
        y_p = y_pred[y_true.eq(lbl_a)]
        labels.each_with_index do |lbl_b, j|
          conf_mat[i, j] = y_p.eq(lbl_b).count
        end
      end

      conf_mat
    end

    # Output a summary of classification performance for each class.
    #
    # @example
    #   y_true = Numo::Int32[0, 1, 1, 2, 2, 2, 0]
    #   y_pred = Numo::Int32[1, 1, 1, 0, 0, 2, 0]
    #   puts Rumale::EvaluationMeasure.classification_report(y_true, y_pred)
    #
    #   #               precision    recall  f1-score   support
    #   #
    #   #            0       0.33      0.50      0.40         2
    #   #            1       0.67      1.00      0.80         2
    #   #            2       1.00      0.33      0.50         3
    #   #
    #   #     accuracy                           0.57         7
    #   #    macro avg       0.67      0.61      0.57         7
    #   # weighted avg       0.71      0.57      0.56         7
    #
    # @param y_true [Numo::Int32] (shape: [n_samples]) The ground truth labels.
    # @param y_pred [Numo::Int32] (shape: [n_samples]) The predicted labels.
    # @param target_name [Nil/Array] The label names.
    # @param output_hash [Boolean] The flag indicating whether to output with Ruby Hash.
    # @return [String/Hash] The summary of classification performance.
    #   If output_hash is true, it returns the summary with Ruby Hash.
    def classification_report(y_true, y_pred, target_name: nil, output_hash: false)
      y_true = Rumale::Validation.check_convert_label_array(y_true)
      y_pred = Rumale::Validation.check_convert_label_array(y_pred)
      # calculate each evaluation measure.
      supports = y_true.bincount
      precisions = Rumale::EvaluationMeasure::PrecisionRecall.precision_each_class(y_true, y_pred)
      recalls = Rumale::EvaluationMeasure::PrecisionRecall.recall_each_class(y_true, y_pred)
      fscores = Rumale::EvaluationMeasure::PrecisionRecall.f_score_each_class(y_true, y_pred)
      macro_precision = Rumale::EvaluationMeasure::PrecisionRecall.macro_average_precision(y_true, y_pred)
      macro_recall = Rumale::EvaluationMeasure::PrecisionRecall.macro_average_recall(y_true, y_pred)
      macro_fscore = Rumale::EvaluationMeasure::PrecisionRecall.macro_average_f_score(y_true, y_pred)
      accuracy = Rumale::EvaluationMeasure::Accuracy.new.score(y_true, y_pred)
      sum_supports = supports.sum
      weights = Numo::DFloat.cast(supports) / sum_supports
      weighted_precision = (Numo::DFloat.cast(precisions) * weights).sum
      weighted_recall = (Numo::DFloat.cast(recalls) * weights).sum
      weighted_fscore = (Numo::DFloat.cast(fscores) * weights).sum
      # output reults.
      target_name ||= y_true.to_a.uniq.sort.map(&:to_s)
      if output_hash
        res = {}
        target_name.each_with_index do |label, n|
          res[label] = {
            precision: precisions[n],
            recall: recalls[n],
            fscore: fscores[n],
            support: supports[n]
          }
        end
        res[:accuracy] = accuracy
        res[:macro_avg] = {
          precision: macro_precision,
          recall: macro_recall,
          fscore: macro_fscore,
          support: sum_supports
        }
        res[:weighted_avg] = {
          precision: weighted_precision,
          recall: weighted_recall,
          fscore: weighted_fscore,
          support: sum_supports
        }
        res
      else
        width = ['weighted avg'.size, target_name.map(&:size).max].max
        res = +''
        res << "#{' ' * width}  precision    recall  f1-score   support\n"
        res << "\n"
        target_name.each_with_index do |label, n|
          label_str = format("%##{width}s", label)
          precision_str = format('%#10s', format('%.2f', precisions[n]))
          recall_str = format('%#10s', format('%.2f', recalls[n]))
          fscore_str = format('%#10s', format('%.2f', fscores[n]))
          supports_str = format('%#10s', supports[n])
          res << "#{label_str} #{precision_str}#{recall_str}#{fscore_str}#{supports_str}\n"
        end
        res << "\n"
        supports_str = format('%#10s', sum_supports)
        accuracy_str = format('%#30s', format('%.2f', accuracy))
        res << format("%##{width}s ", 'accuracy')
        res << "#{accuracy_str}#{supports_str}\n"
        precision_str = format('%#10s', format('%.2f', macro_precision))
        recall_str = format('%#10s', format('%.2f', macro_recall))
        fscore_str = format('%#10s', format('%.2f', macro_fscore))
        res << format("%##{width}s ", 'macro avg')
        res << "#{precision_str}#{recall_str}#{fscore_str}#{supports_str}\n"
        precision_str = format('%#10s', format('%.2f', weighted_precision))
        recall_str = format('%#10s', format('%.2f', weighted_recall))
        fscore_str = format('%#10s', format('%.2f', weighted_fscore))
        res << format("%##{width}s ", 'weighted avg')
        res << "#{precision_str}#{recall_str}#{fscore_str}#{supports_str}\n"
        res
      end
    end
  end
end
