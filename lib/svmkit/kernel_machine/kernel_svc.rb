# frozen_string_literal: true

require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implement kernel method-based estimator.
  module KernelMachine
    # KernelSVC is a class that implements (Nonlinear) Kernel Support Vector Classifier
    # with stochastic gradient descent (SGD) optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # @example
    #   training_kernel_matrix = SVMKit::PairwiseMetric::rbf_kernel(training_samples)
    #   estimator =
    #     SVMKit::KernelMachine::KernelSVC.new(reg_param: 1.0, max_iter: 1000, random_seed: 1)
    #   estimator.fit(training_kernel_matrix, traininig_labels)
    #   testing_kernel_matrix = SVMKit::PairwiseMetric::rbf_kernel(testing_samples, training_samples)
    #   results = estimator.predict(testing_kernel_matrix)
    #
    # *Reference*
    # 1. S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Mathematical Programming, vol. 127 (1), pp. 3--30, 2011.
    class KernelSVC
      include Base::BaseEstimator
      include Base::Classifier

      # Return the weight vector for Kernel SVC.
      # @return [Numo::DFloat] (shape: [n_classes, n_trainig_sample])
      attr_reader :weight_vec

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Return the random generator for performing random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Kernel Support Vector Machine by the SGD optimization.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, max_iter: 1000, random_seed: nil)
        SVMKit::Validation.check_params_float(reg_param: reg_param)
        SVMKit::Validation.check_params_integer(max_iter: max_iter)
        SVMKit::Validation.check_params_type_or_nil(Integer, random_seed: random_seed)
        SVMKit::Validation.check_params_positive(reg_param: reg_param, max_iter: max_iter)
        @params = {}
        @params[:reg_param] = reg_param
        @params[:max_iter] = max_iter
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @weight_vec = nil
        @classes = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_training_samples, n_training_samples])
      #   The kernel matrix of the training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_training_samples]) The labels to be used for fitting the model.
      # @return [KernelSVC] The learned classifier itself.
      def fit(x, y)
        SVMKit::Validation.check_sample_array(x)
        SVMKit::Validation.check_label_array(y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_classes = @classes.size
        _n_samples, n_features = x.shape

        if n_classes > 2
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          n_classes.times do |n|
            bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
            @weight_vec[n, true] = binary_fit(x, bin_y)
          end
        else
          negative_label = y.to_a.uniq.sort.first
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec = binary_fit(x, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_testing_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        SVMKit::Validation.check_sample_array(x)

        x.dot(@weight_vec.transpose)
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_testing_samples]) Predicted class label per sample.
      def predict(x)
        SVMKit::Validation.check_sample_array(x)

        return Numo::Int32.cast(decision_function(x).ge(0.0)) * 2 - 1 if @classes.size <= 2

        n_samples, = x.shape
        decision_values = decision_function(x)
        Numo::Int32.asarray(Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] })
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about KernelSVC.
      def marshal_dump
        { params: @params,
          weight_vec: @weight_vec,
          classes: @classes,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @weight_vec = obj[:weight_vec]
        @classes = obj[:classes]
        @rng = obj[:rng]
        nil
      end

      private

      def binary_fit(x, bin_y)
        # Initialize some variables.
        n_training_samples = x.shape[0]
        rand_ids = []
        weight_vec = Numo::DFloat.zeros(n_training_samples)
        # Start optimization.
        @params[:max_iter].times do |t|
          # random sampling
          rand_ids = [*0...n_training_samples].shuffle(random: @rng) if rand_ids.empty?
          target_id = rand_ids.shift
          # update the weight vector
          func = (weight_vec * bin_y[target_id]).dot(x[target_id, true].transpose).to_f
          func *= bin_y[target_id] / (@params[:reg_param] * (t + 1))
          weight_vec[target_id] += 1.0 if func < 1.0
        end
        weight_vec * Numo::DFloat[*bin_y]
      end
    end
  end
end
