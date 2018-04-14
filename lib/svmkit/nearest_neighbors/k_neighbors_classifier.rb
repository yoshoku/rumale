# frozen_string_literal: true

require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implement estimators based on nearest neighbors rule.
  module NearestNeighbors
    # KNeighborsClassifier is a class that implements the classifier with the k-nearest neighbors rule.
    # The current implementation uses the Euclidean distance for finding the neighbors.
    #
    # @example
    #   estimator =
    #     SVMKit::NearestNeighbor::KNeighborsClassifier.new(n_neighbors = 5)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    class KNeighborsClassifier
      include Base::BaseEstimator
      include Base::Classifier

      # Return the prototypes for the nearest neighbor classifier.
      # @return [Numo::DFloat] (shape: [n_samples, n_features])
      attr_reader :prototypes

      # Return the labels of the prototypes
      # @return [Numo::Int32] (size: n_samples)
      attr_reader :labels

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Create a new classifier with the nearest neighbor rule.
      #
      # @param n_neighbors [Integer] The number of neighbors.
      def initialize(n_neighbors: 5)
        SVMKit::Validation.check_params_integer(n_neighbors: n_neighbors)
        SVMKit::Validation.check_params_positive(n_neighbors: n_neighbors)
        @params = {}
        @params[:n_neighbors] = n_neighbors
        @prototypes = nil
        @labels = nil
        @classes = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [KNeighborsClassifier] The learned classifier itself.
      def fit(x, y)
        SVMKit::Validation.check_sample_array(x)
        SVMKit::Validation.check_label_array(y)
        @prototypes = Numo::DFloat.asarray(x.to_a)
        @labels = Numo::Int32.asarray(y.to_a)
        @classes = Numo::Int32.asarray(y.to_a.uniq.sort)
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence scores per sample for each class.
      def decision_function(x)
        SVMKit::Validation.check_sample_array(x)
        distance_matrix = PairwiseMetric.euclidean_distance(x, @prototypes)
        n_samples, n_prototypes = distance_matrix.shape
        n_classes = @classes.size
        n_neighbors = [@params[:n_neighbors], n_prototypes].min
        scores = Numo::DFloat.zeros(n_samples, n_classes)
        n_samples.times do |m|
          neighbor_ids = distance_matrix[m, true].to_a.each_with_index.sort.map(&:last)[0...n_neighbors]
          neighbor_ids.each { |n| scores[m, @classes.to_a.index(@labels[n])] += 1.0 }
        end
        scores
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        SVMKit::Validation.check_sample_array(x)
        n_samples = x.shape.first
        decision_values = decision_function(x)
        Numo::Int32.asarray(Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] })
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about KNeighborsClassifier.
      def marshal_dump
        { params: @params,
          prototypes: @prototypes,
          labels: @labels,
          classes: @classes }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @prototypes = obj[:prototypes]
        @labels = obj[:labels]
        @classes = obj[:classes]
        nil
      end
    end
  end
end
