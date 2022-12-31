# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/classifier'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  # This module consists of the classes that implement estimators based on nearest neighbors rule.
  module NearestNeighbors
    # KNeighborsClassifier is a class that implements the classifier with the k-nearest neighbors rule.
    # The current implementation uses the Euclidean distance for finding the neighbors.
    #
    # @example
    #   require 'rumale/nearest_neighbors/k_neighbors_classifier'
    #
    #   estimator =
    #     Rumale::NearestNeighbors::KNeighborsClassifier.new(n_neighbors: 5)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    class KNeighborsClassifier < ::Rumale::Base::Estimator
      include ::Rumale::Base::Classifier

      # Return the prototypes for the nearest neighbor classifier.
      # If the metric is 'precomputed', that returns nil.
      # @return [Numo::DFloat] (shape: [n_training_samples, n_features])
      attr_reader :prototypes

      # Return the labels of the prototypes
      # @return [Numo::Int32] (size: n_training_samples)
      attr_reader :labels

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Create a new classifier with the nearest neighbor rule.
      #
      # @param n_neighbors [Integer] The number of neighbors.
      # @param metric [String] The metric to calculate the distances.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance between points.
      #   If metric is 'precomputed', the fit and predict methods expect to be given a distance matrix.
      def initialize(n_neighbors: 5, metric: 'euclidean')
        super()
        @params = {
          n_neighbors: n_neighbors,
          metric: (metric == 'precomputed' ? 'precomputed' : 'euclidean')
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_training_samples, n_features]) The training data to be used for fitting the model.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_training_samples, n_training_samples]).
      # @param y [Numo::Int32] (shape: [n_training_samples]) The labels to be used for fitting the model.
      # @return [KNeighborsClassifier] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
          raise ArgumentError, 'Expect the input distance matrix to be square.'
        end

        @prototypes = x.dup if @params[:metric] == 'euclidean'
        @labels = Numo::Int32.asarray(y.to_a)
        @classes = Numo::Int32.asarray(y.to_a.uniq.sort)
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_features]) The samples to compute the scores.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_testing_samples, n_training_samples]).
      # @return [Numo::DFloat] (shape: [n_testing_samples, n_classes]) Confidence scores per sample for each class.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        if @params[:metric] == 'precomputed' && x.shape[1] != @labels.size
          raise ArgumentError, 'Expect the size input matrix to be n_testing_samples-by-n_training_samples.'
        end

        n_prototypes = @labels.size
        n_neighbors = [@params[:n_neighbors], n_prototypes].min
        n_samples = x.shape[0]
        n_classes = @classes.size
        scores = Numo::DFloat.zeros(n_samples, n_classes)

        distance_matrix = @params[:metric] == 'precomputed' ? x : ::Rumale::PairwiseMetric.euclidean_distance(x, @prototypes)
        n_samples.times do |m|
          neighbor_ids = distance_matrix[m, true].to_a.each_with_index.sort.map(&:last)[0...n_neighbors]
          neighbor_ids.each { |n| scores[m, @classes.to_a.index(@labels[n])] += 1.0 }
        end

        scores
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_features]) The samples to predict the labels.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_testing_samples, n_training_samples]).
      # @return [Numo::Int32] (shape: [n_testing_samples]) Predicted class label per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        if @params[:metric] == 'precomputed' && x.shape[1] != @labels.size
          raise ArgumentError, 'Expect the size input matrix to be n_samples-by-n_training_samples.'
        end

        decision_values = decision_function(x)
        n_samples = x.shape[0]
        Numo::Int32.asarray(Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] })
      end
    end
  end
end
