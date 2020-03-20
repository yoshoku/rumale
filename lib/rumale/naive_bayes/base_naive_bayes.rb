# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/classifier'

module Rumale
  # This module consists of the classes that implement naive bayes models.
  module NaiveBayes
    # BaseNaiveBayes is a class that has methods for common processes of naive bayes classifier.
    # This class is used internally.
    class BaseNaiveBayes
      include Base::BaseEstimator
      include Base::Classifier

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        n_samples = x.shape.first
        decision_values = decision_function(x)
        Numo::Int32.asarray(Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] })
      end

      # Predict log-probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the log-probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted log-probability of each class per sample.
      def predict_log_proba(x)
        x = check_convert_sample_array(x)
        n_samples, = x.shape
        log_likelihoods = decision_function(x)
        log_likelihoods - Numo::NMath.log(Numo::NMath.exp(log_likelihoods).sum(1)).reshape(n_samples, 1)
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = check_convert_sample_array(x)
        Numo::NMath.exp(predict_log_proba(x)).abs
      end
    end
  end
end
