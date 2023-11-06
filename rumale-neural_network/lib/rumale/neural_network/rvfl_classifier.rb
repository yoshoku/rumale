# frozen_string_literal: true

require 'rumale/base/classifier'
require 'rumale/neural_network/base_rvfl'
require 'rumale/utils'
require 'rumale/validation'

module Rumale
  module NeuralNetwork
    # RVFLClassifier is a class that implements classifier based on random vector functional link (RVFL) network.
    # The current implementation uses sigmoid function as activation function.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #   Numo::Linalg = Numo::TinyLinalg
    #
    #   require 'rumale/neural_network/rvfl_classifier'
    #
    #   estimator = Rumale::NeuralNetwork::RVFLClassifier.new(hidden_units: 128, reg_param: 100.0)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Malik, A. K., Gao, R., Ganaie, M. A., Tanveer, M., and Suganthan, P. N., "Random vector functional link network: recent developments, applications, and future directions," Applied Soft Computing, vol. 143, 2023.
    # - Zhang, L., and Suganthan, P. N., "A comprehensive evaluation of random vector functional link networks," Information Sciences, vol. 367--368, pp. 1094--1105, 2016.
    class RVFLClassifier < BaseRVFL
      include ::Rumale::Base::Classifier

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the weight vector in the hidden layer of RVFL network.
      # @return [Numo::DFloat] (shape: [n_hidden_units, n_features])
      attr_reader :random_weight_vec

      # Return the bias vector in the hidden layer of RVFL network.
      # @return [Numo::DFloat] (shape: [n_hidden_units])
      attr_reader :random_bias

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_features + n_hidden_units, n_classes])
      attr_reader :weight_vec

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with RVFL network.
      #
      # @param hidden_units [Array] The number of units in the hidden layer.
      # @param reg_param [Float] The regularization parameter.
      # @param scale [Float] The scale parameter for random weight and bias.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(hidden_units: 128, reg_param: 100.0, scale: 1.0, random_seed: nil)
        super
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [RVFLClassifier] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        raise 'RVFLClassifier#fit requires Numo::Linalg but that is not loaded.' unless enable_linalg?(warning: false)

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
