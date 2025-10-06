# frozen_string_literal: true

require 'rumale/base/classifier'
require 'rumale/utils'
require 'rumale/validation'
require 'rumale/neural_network/base_rbf'

module Rumale
  module NeuralNetwork
    # RBFClassifier is a class that implements classifier based on (k-means) radial basis function (RBF) networks.
    #
    # @example
    #   require 'numo/linalg'
    #   require 'rumale/neural_network/rbf_classifier'
    #
    #   estimator = Rumale::NeuralNetwork::RBFClassifier.new(hidden_units: 128, reg_param: 100.0)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Bugmann, G., "Normalized Gaussian Radial Basis Function networks," Neural Computation, vol. 20, pp. 97--110, 1998.
    # - Que, Q., and Belkin, M., "Back to the Future: Radial Basis Function Networks Revisited," Proc. of AISTATS'16, pp. 1375--1383, 2016.
    class RBFClassifier < BaseRBF
      include ::Rumale::Base::Classifier

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the centers in the hidden layer of RBF network.
      # @return [Numo::DFloat] (shape: [n_centers, n_features])
      attr_reader :centers

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_centers, n_classes])
      attr_reader :weight_vec

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with (k-means) RBF networks.
      #
      # @param hidden_units [Array] The number of units in the hidden layer.
      # @param gamma [Float] The parameter for the radial basis function, if nil it is 1 / n_features.
      # @param reg_param [Float] The regularization parameter.
      # @param normalize [Boolean] The flag indicating whether to normalize the hidden layer output or not.
      # @param max_iter [Integer] The maximum number of iterations for finding centers.
      # @param tol [Float] The tolerance of termination criterion for finding centers.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(hidden_units: 128, gamma: nil, reg_param: 100.0, normalize: false,
                     max_iter: 50, tol: 1e-4, random_seed: nil)
        super
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [RBFClassifier] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        raise 'RBFClassifier#fit requires Numo::Linalg but that is not loaded.' unless enable_linalg?(warning: false)

        @classes = Numo::NArray[*y.to_a.uniq.sort]

        partial_fit(x, one_hot_encode(y))

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        h = hidden_output(x)
        h.dot(@weight_vec)
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        scores = decision_function(x)
        n_samples, n_classes = scores.shape
        label_ids = scores.max_index(axis: 1) - Numo::Int32.new(n_samples).seq * n_classes
        @classes[label_ids].dup
      end

      private

      def one_hot_encode(y)
        Numo::DFloat.cast(::Rumale::Utils.binarize_labels(y))
      end
    end
  end
end
