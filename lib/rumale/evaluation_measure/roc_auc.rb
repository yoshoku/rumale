# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # ROCAUC is a class that calculate area under the receiver operation characteristic curve from predicted scores.
    #
    # @example
    #   # Encode labels to integer array.
    #   labels = %w[A B B C A A C C C A]
    #   label_encoder = Rumale::Preprocessing::LabelEncoder.new
    #   y = label_encoder.fit_transform(labels)
    #   # Fit classifier.
    #   classifier = Rumale::LinearModel::LogisticRegression.new
    #   classifier.fit(x, y)
    #   # Predict class probabilities.
    #   y_score = classifier.predict_proba(x)
    #   # Encode labels to one-hot vectors.
    #   one_hot_encoder = Rumale::Preprocessing::OneHotEncoder.new
    #   y_onehot = one_hot_encoder.fit_transform(y)
    #   # Calculate ROC AUC.
    #   evaluator = Rumale::EvaluationMeasure::ROCAUC.new
    #   puts evaluator.score(y_onehot, y_score)
    class ROCAUC
      include Base::Evaluator

      # Calculate area under the receiver operation characteristic curve (ROC AUC).
      #
      # @param y_true [Numo::Int32] (shape: [n_samples] or [n_samples, n_classes])
      #   Ground truth binary labels or one-hot encoded multi-labels.
      # @param y_score [Numo::DFloat] (shape: [n_samples] or [n_samples, n_classes])
      #   Predicted class probabilities or confidence scores.
      # @return [Float] (macro-averaged) ROC AUC.
      def score(y_true, y_score)
        y_true = Numo::Int32.cast(y_true) unless y_true.is_a?(Numo::Int32)
        y_score = Numo::DFloat.cast(y_score) unless y_score.is_a?(Numo::DFloat)
        raise ArgumentError, 'Expect to have the same shape for y_true and y_score.' unless y_true.shape == y_score.shape

        n_classes = y_score.shape[1]
        if n_classes.nil?
          fpr, tpr, = roc_curve(y_true, y_score)
          return auc(fpr, tpr)
        end

        scores = Array.new(n_classes) do |c|
          fpr, tpr, = roc_curve(y_true[true, c], y_score[true, c])
          auc(fpr, tpr)
        end

        scores.reduce(&:+).fdiv(n_classes)
      end

      # Calculate receiver operation characteristic curve.
      #
      # @param y_true [Numo::Int32] (shape: [n_samples]) Ground truth binary labels.
      # @param y_score [Numo::DFloat] (shape: [n_samples]) Predicted class probabilities or confidence scores.
      # @param pos_label [Integer] Label to be a positive label when binarizing the given labels.
      #   If nil is given, the method considers the maximum value of the label as a positive label.
      # @return [Array] fpr (Numo::DFloat): false positive rates. tpr (Numo::DFloat): true positive rates.
      #   thresholds (Numo::DFloat): thresholds on the decision function used to calculate fpr and tpr.
      def roc_curve(y_true, y_score, pos_label = nil)
        y_true = Numo::Int32.cast(y_true) unless y_true.is_a?(Numo::Int32)
        y_score = Numo::DFloat.cast(y_score) unless y_score.is_a?(Numo::DFloat)
        raise ArgumentError, 'Expect y_true to be 1-D arrray.' unless y_true.shape[1].nil?
        raise ArgumentError, 'Expect y_score to be 1-D arrray.' unless y_score.shape[1].nil?

        labels = y_true.to_a.uniq
        if pos_label.nil?
          raise ArgumentError, 'y_true must be binary labels or pos_label must be specified if y_true is multi-label' unless labels.size == 2
        else
          raise ArgumentError, 'y_true must have elements whose values are pos_label.' unless y_true.to_a.uniq.include?(pos_label)
        end

        false_pos, true_pos, thresholds = binary_roc_curve(y_true, y_score, pos_label)

        if true_pos.size.zero? || false_pos[0] != 0 || true_pos[0] != 0
          true_pos = true_pos.insert(0, 0)
          false_pos = false_pos.insert(0, 0)
          thresholds = thresholds.insert(0, thresholds[0] + 1)
        end

        tpr = true_pos / true_pos[-1].to_f
        fpr = false_pos / false_pos[-1].to_f

        [fpr, tpr, thresholds]
      end

      # Calculate area under the curve using the trapezoidal rule.
      #
      # @param x [Numo::Int32/Numo::DFloat] (shape: [n_elements])
      #   x coordinates. These are expected to monotonously increase or decrease.
      # @param y [Numo::Int32/Numo::DFloat] (shape: [n_elements]) y coordinates.
      # @return [Float] area under the curve.
      def auc(x, y)
        x = Numo::NArray.asarray(x) unless x.is_a?(Numo::NArray)
        y = Numo::NArray.asarray(y) unless y.is_a?(Numo::NArray)
        raise ArgumentError, 'Expect x to be 1-D arrray.' unless x.shape[1].nil?
        raise ArgumentError, 'Expect y to be 1-D arrray.' unless y.shape[1].nil?

        n_samples = [x.shape[0], y.shape[0]].min
        raise ArgumentError, 'At least two points are required to calculate area under curve.' if n_samples < 2

        (0...n_samples).to_a.each_cons(2).map { |i, j| 0.5 * (x[i] - x[j]).abs * (y[i] + y[j]) }.reduce(&:+)
      end

      private

      def binary_roc_curve(y_true, y_score, pos_label = nil)
        pos_label = y_true.to_a.uniq.max if pos_label.nil?

        bin_y_true = y_true.eq(pos_label)
        desc_pred_ids = y_score.sort_index.reverse

        desc_y_true = Numo::Int32.cast(bin_y_true[desc_pred_ids])
        desc_y_score = y_score[desc_pred_ids]

        dist_value_ids = desc_y_score.diff.ne(0).where
        threshold_ids = dist_value_ids.append(desc_y_true.size - 1)

        true_pos = desc_y_true.cumsum[threshold_ids]
        false_pos = 1 + threshold_ids - true_pos

        [false_pos, true_pos, desc_y_score[threshold_ids]]
      end
    end
  end
end
