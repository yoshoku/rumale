# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/classifier'
require 'rumale/preprocessing/label_binarizer'

module Rumale
  module KernelMachine
    # KernelRidgeClassifier is a class that implements classifier based-on kernel ridge regression.
    # It learns a classifier by converting labels to target values { -1, 1 } and performing kernel ridge regression.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale'
    #
    #   kernel_mat_train = Rumale::PairwiseMetric::rbf_kernel(training_samples)
    #   kridge = Rumale::KernelMachine::KernelRidgeClassifier.new(reg_param: 0.5)
    #   kridge.fit(kernel_mat_train, traininig_values)
    #
    #   kernel_mat_test = Rumale::PairwiseMetric::rbf_kernel(test_samples, training_samples)
    #   results = kridge.predict(kernel_mat_test)
    class KernelRidgeClassifier
      include Base::BaseEstimator
      include Base::Classifier

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_training_sample, n_classes])
      attr_reader :weight_vec

      # Create a new regressor with kernel ridge classifier.
      #
      # @param reg_param [Float/Numo::DFloat] The regularization parameter.
      def initialize(reg_param: 1.0)
        @params = {}
        @params[:reg_param] = reg_param
        @classes = nil
        @weight_vec = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_training_samples, n_training_samples])
      #   The kernel matrix of the training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_training_samples]) The labels to be used for fitting the model.
      # @return [KernelRidgeClassifier] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)
        raise ArgumentError, 'Expect the kernel matrix of training data to be square.' unless x.shape[0] == x.shape[1]
        raise 'KernelRidgeClassifier#fit requires Numo::Linalg but that is not loaded.' unless enable_linalg?

        @encoder = Rumale::Preprocessing::LabelBinarizer.new
        y_encoded = Numo::DFloat.cast(@encoder.fit_transform(y)) * 2 - 1
        @classes = Numo::NArray[*@encoder.classes]

        n_samples = x.shape[0]
        reg_kernel_mat = x + Numo::DFloat.eye(n_samples) * @params[:reg_param]
        @weight_vec = Numo::Linalg.solve(reg_kernel_mat, y_encoded, driver: 'sym')

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to predict values.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) The confidence score per sample.
      def decision_function(x)
        x = check_convert_sample_array(x)
        x.dot(@weight_vec)
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_testing_samples]) Predicted class label per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        scores = decision_function(x)
        n_samples, n_classes = scores.shape
        label_ids = scores.max_index(axis: 1) - Numo::Int32.new(n_samples).seq * n_classes
        @classes[label_ids].dup
      end
    end
  end
end
