# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/classifier'
require 'rumale/probabilistic_output'

module Rumale
  # This module consists of the classes that implement kernel method-based estimator.
  module KernelMachine
    # KernelSVC is a class that implements (Nonlinear) Kernel Support Vector Classifier
    # with stochastic gradient descent (SGD) optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # @note
    #   Rumale::SVM provides kernel support vector classifier based on LIBSVM.
    #   If you prefer execution speed, you should use Rumale::SVM::SVC.
    #   https://github.com/yoshoku/rumale-svm
    #
    # @example
    #   training_kernel_matrix = Rumale::PairwiseMetric::rbf_kernel(training_samples)
    #   estimator =
    #     Rumale::KernelMachine::KernelSVC.new(reg_param: 1.0, max_iter: 1000, random_seed: 1)
    #   estimator.fit(training_kernel_matrix, traininig_labels)
    #   testing_kernel_matrix = Rumale::PairwiseMetric::rbf_kernel(testing_samples, training_samples)
    #   results = estimator.predict(testing_kernel_matrix)
    #
    # *Reference*
    # - Shalev-Shwartz, S., Singer, Y., Srebro, N., and Cotter, A., "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Mathematical Programming, vol. 127 (1), pp. 3--30, 2011.
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
      # @param probability [Boolean] The flag indicating whether to perform probability estimation.
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, max_iter: 1000, probability: false, n_jobs: nil, random_seed: nil)
        check_params_numeric(reg_param: reg_param, max_iter: max_iter)
        check_params_boolean(probability: probability)
        check_params_numeric_or_nil(n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(reg_param: reg_param, max_iter: max_iter)
        @params = {}
        @params[:reg_param] = reg_param
        @params[:max_iter] = max_iter
        @params[:probability] = probability
        @params[:n_jobs] = n_jobs
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @weight_vec = nil
        @prob_param = nil
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
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_classes = @classes.size
        n_features = x.shape[1]

        if n_classes > 2
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @prob_param = Numo::DFloat.zeros(n_classes, 2)
          models = if enable_parallel?
                     # :nocov:
                     parallel_map(n_classes) do |n|
                       bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
                       partial_fit(x, bin_y)
                     end
                     # :nocov:
                   else
                     Array.new(n_classes) do |n|
                       bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
                       partial_fit(x, bin_y)
                     end
                   end
          models.each_with_index { |model, n| @weight_vec[n, true], @prob_param[n, true] = model }
        else
          negative_label = y.to_a.uniq.min
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @prob_param = partial_fit(x, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_testing_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        x = check_convert_sample_array(x)

        x.dot(@weight_vec.transpose)
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_testing_samples]) Predicted class label per sample.
      def predict(x)
        x = check_convert_sample_array(x)

        return Numo::Int32.cast(decision_function(x).ge(0.0)) * 2 - 1 if @classes.size <= 2

        n_samples, = x.shape
        decision_values = decision_function(x)
        predicted = if enable_parallel?
                      parallel_map(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                    else
                      Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                    end
        Numo::Int32.asarray(predicted)
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to predict the labels.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = check_convert_sample_array(x)

        if @classes.size > 2
          probs = 1.0 / (Numo::NMath.exp(@prob_param[true, 0] * decision_function(x) + @prob_param[true, 1]) + 1.0)
          return (probs.transpose / probs.sum(axis: 1)).transpose.dup
        end

        n_samples, = x.shape
        probs = Numo::DFloat.zeros(n_samples, 2)
        probs[true, 1] = 1.0 / (Numo::NMath.exp(@prob_param[0] * decision_function(x) + @prob_param[1]) + 1.0)
        probs[true, 0] = 1.0 - probs[true, 1]
        probs
      end

      private

      def partial_fit(x, bin_y)
        # Initialize some variables.
        n_training_samples = x.shape[0]
        rand_ids = []
        weight_vec = Numo::DFloat.zeros(n_training_samples)
        sub_rng = @rng.dup
        # Start optimization.
        @params[:max_iter].times do |t|
          # random sampling
          rand_ids = Array(0...n_training_samples).shuffle(random: sub_rng) if rand_ids.empty?
          target_id = rand_ids.shift
          # update the weight vector
          func = (weight_vec * bin_y).dot(x[target_id, true].transpose).to_f
          func *= bin_y[target_id] / (@params[:reg_param] * (t + 1))
          weight_vec[target_id] += 1.0 if func < 1.0
        end
        w = weight_vec * bin_y
        p = if @params[:probability]
              Rumale::ProbabilisticOutput.fit_sigmoid(x.dot(w), bin_y)
            else
              Numo::DFloat[1, 0]
            end
        [w, p]
      end
    end
  end
end
