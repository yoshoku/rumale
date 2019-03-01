# frozen_string_literal: true

require 'rumale/base/base_estimator.rb'
require 'rumale/base/classifier.rb'

module Rumale
  # This module consists of the classes that implement multi-class classification strategy.
  module Multiclass
    # @note
    #   All classifier in Rumale support multi-class classifiction since version 0.2.7.
    #   There is no need to explicitly use this class for multiclass classifiction.
    #
    # OneVsRestClassifier is a class that implements One-vs-Rest (OvR) strategy for multi-class classification.
    #
    # @example
    #   base_estimator = Rumale::LinearModel::LogisticRegression.new
    #   estimator = Rumale::Multiclass::OneVsRestClassifier.new(estimator: base_estimator)
    #   estimator.fit(training_samples, training_labels)
    #   results = estimator.predict(testing_samples)
    class OneVsRestClassifier
      include Base::BaseEstimator
      include Base::Classifier

      # Return the set of estimators.
      # @return [Array<Classifier>]
      attr_reader :estimators

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Create a new multi-class classifier with the one-vs-rest startegy.
      #
      # @param estimator [Classifier] The (binary) classifier for construction a multi-class classifier.
      def initialize(estimator: nil)
        check_params_type(Rumale::Base::BaseEstimator, estimator: estimator)
        @params = {}
        @params[:estimator] = estimator
        @estimators = nil
        @classes = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [OneVsRestClassifier] The learned classifier itself.
      def fit(x, y)
        check_sample_array(x)
        check_label_array(y)
        check_sample_label_size(x, y)
        y_arr = y.to_a
        @classes = Numo::Int32.asarray(y_arr.uniq.sort)
        @estimators = @classes.to_a.map do |label|
          bin_y = Numo::Int32.asarray(y_arr.map { |l| l == label ? 1 : -1 })
          @params[:estimator].dup.fit(x, bin_y)
        end
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence scores per sample for each class.
      def decision_function(x)
        check_sample_array(x)
        n_classes = @classes.size
        Numo::DFloat.asarray(Array.new(n_classes) { |m| @estimators[m].decision_function(x).to_a }).transpose
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        check_sample_array(x)
        n_samples, = x.shape
        decision_values = decision_function(x)
        Numo::Int32.asarray(Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] })
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about OneVsRestClassifier.
      def marshal_dump
        { params: @params,
          classes: @classes,
          estimators: @estimators.map { |e| Marshal.dump(e) } }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @classes = obj[:classes]
        @estimators = obj[:estimators].map { |e| Marshal.load(e) }
        nil
      end
    end
  end
end
