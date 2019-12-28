# frozen_string_literal: true

require 'rumale/linear_model/base_linear_model'
require 'rumale/base/classifier'
require 'rumale/probabilistic_output'

module Rumale
  # This module consists of the classes that implement generalized linear models.
  module LinearModel
    # SVC is a class that implements Support Vector Classifier
    # with mini-batch stochastic gradient descent optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # Rumale::SVM provides linear support vector classifier based on LIBLINEAR.
    # If you prefer execution speed, you should use Rumale::SVM::LinearSVC.
    # https://github.com/yoshoku/rumale-svm
    #
    # @example
    #   estimator =
    #     Rumale::LinearModel::SVC.new(reg_param: 1.0, max_iter: 1000, batch_size: 20, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Shalev-Shwartz and Y. Singer, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Proc. ICML'07, pp. 807--814, 2007.
    class SVC < BaseLinearModel
      include Base::Classifier

      # Return the weight vector for SVC.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for SVC.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :bias_term

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Return the random generator for performing random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Support Vector Machine by the SGD optimization.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param probability [Boolean] The flag indicating whether to perform probability estimation.
      # @param optimizer [Optimizer] The optimizer to calculate adaptive learning rate.
      #   If nil is given, Nadam is used.
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, fit_bias: false, bias_scale: 1.0,
                     max_iter: 1000, batch_size: 20, probability: false, optimizer: nil, n_jobs: nil, random_seed: nil)
        check_params_numeric(reg_param: reg_param, bias_scale: bias_scale, max_iter: max_iter, batch_size: batch_size)
        check_params_boolean(fit_bias: fit_bias, probability: probability)
        check_params_numeric_or_nil(n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(reg_param: reg_param, bias_scale: bias_scale, max_iter: max_iter, batch_size: batch_size)
        keywd_args = method(:initialize).parameters.map { |_t, arg| [arg, binding.local_variable_get(arg)] }.to_h
        keywd_args.delete(:probability)
        super(**keywd_args)
        @params[:probability] = probability
        @prob_param = nil
        @classes = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [SVC] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]

        if multiclass_problem?
          n_classes = @classes.size
          n_features = x.shape[1]
          # initialize model.
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          @prob_param = Numo::DFloat.zeros(n_classes, 2)
          # fit model.
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
          # store model.
          models.each_with_index { |model, n| @weight_vec[n, true], @bias_term[n], @prob_param[n, true] = model }
        else
          negative_label = @classes[0]
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @bias_term, @prob_param = partial_fit(x, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        x = check_convert_sample_array(x)
        x.dot(@weight_vec.transpose) + @bias_term
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = check_convert_sample_array(x)

        n_samples = x.shape[0]
        predicted = if multiclass_problem?
                      decision_values = decision_function(x)
                      if enable_parallel?
                        parallel_map(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                      else
                        Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                      end
                    else
                      decision_values = decision_function(x).ge(0.0).to_a
                      Array.new(n_samples) { |n| @classes[decision_values[n]] }
                    end
        Numo::Int32.asarray(predicted)
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = check_convert_sample_array(x)

        if multiclass_problem?
          probs = 1.0 / (Numo::NMath.exp(@prob_param[true, 0] * decision_function(x) + @prob_param[true, 1]) + 1.0)
          (probs.transpose / probs.sum(axis: 1)).transpose.dup
        else
          n_samples, = x.shape
          probs = Numo::DFloat.zeros(n_samples, 2)
          probs[true, 1] = 1.0 / (Numo::NMath.exp(@prob_param[0] * decision_function(x) + @prob_param[1]) + 1.0)
          probs[true, 0] = 1.0 - probs[true, 1]
          probs
        end
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about SVC.
      def marshal_dump
        { params: @params,
          weight_vec: @weight_vec,
          bias_term: @bias_term,
          prob_param: @prob_param,
          classes: @classes,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        @prob_param = obj[:prob_param]
        @classes = obj[:classes]
        @rng = obj[:rng]
        nil
      end

      private

      def partial_fit(x, bin_y)
        w, b = super
        p = if @params[:probability]
              Rumale::ProbabilisticOutput.fit_sigmoid(x.dot(w.transpose) + b, bin_y)
            else
              Numo::DFloat[1, 0]
            end
        [w, b, p]
      end

      def calc_loss_gradient(x, y, weight)
        target_ids = (x.dot(weight) * y).lt(1.0).where
        grad = Numo::DFloat.zeros(@params[:batch_size])
        grad[target_ids] = -y[target_ids]
        grad
      end

      def multiclass_problem?
        @classes.size > 2
      end
    end
  end
end
