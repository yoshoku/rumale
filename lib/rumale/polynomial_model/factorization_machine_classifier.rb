# frozen_string_literal: true

require 'rumale/base/classifier'
require 'rumale/polynomial_model/base_factorization_machine'

module Rumale
  # This module consists of the classes that implement polynomial models.
  module PolynomialModel
    # FactorizationMachineClassifier is a class that implements Factorization Machine
    # with stochastic gradient descent (SGD) optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # @example
    #   estimator =
    #     Rumale::PolynomialModel::FactorizationMachineClassifier.new(
    #      n_factors: 10, loss: 'hinge', reg_param_linear: 0.001, reg_param_factor: 0.001,
    #      max_iter: 500, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Rendle, S., "Factorization Machines with libFM," ACM TIST, vol. 3 (3), pp. 57:1--57:22, 2012.
    # - Rendle, S., "Factorization Machines," Proc. ICDM'10, pp. 995--1000, 2010.
    class FactorizationMachineClassifier < BaseFactorizationMachine
      include Base::Classifier

      # Return the factor matrix for Factorization Machine.
      # @return [Numo::DFloat] (shape: [n_classes, n_factors, n_features])
      attr_reader :factor_mat

      # Return the weight vector for Factorization Machine.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :weight_vec

      # Return the bias term for Factoriazation Machine.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :bias_term

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Return the random generator for random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Factorization Machine.
      #
      # @param n_factors [Integer] The maximum number of iterations.
      # @param loss [String] The loss function ('hinge' or 'logistic').
      # @param reg_param_linear [Float] The regularization parameter for linear model.
      # @param reg_param_factor [Float] The regularization parameter for factor matrix.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param batch_size [Integer] The size of the mini batches.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      # @param optimizer [Optimizer] The optimizer to calculate adaptive learning rate.
      #   If nil is given, Nadam is used.
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2, loss: 'hinge', reg_param_linear: 1.0, reg_param_factor: 1.0,
                     max_iter: 200, batch_size: 50, tol: 1e-4,
                     optimizer: nil, n_jobs: nil, verbose: false, random_seed: nil)
        check_params_numeric(reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor,
                             n_factors: n_factors, max_iter: max_iter, batch_size: batch_size, tol: tol)
        check_params_string(loss: loss)
        check_params_boolean(verbose: verbose)
        check_params_numeric_or_nil(n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(n_factors: n_factors,
                              reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor,
                              max_iter: max_iter, batch_size: batch_size)
        super
        @classes = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [FactorizationMachineClassifier] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]

        if multiclass_problem?
          n_classes = @classes.size
          n_features = x.shape[1]
          @factor_mat = Numo::DFloat.zeros(n_classes, @params[:n_factors], n_features)
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          if enable_parallel?
            # :nocov:
            models = parallel_map(n_classes) do |n|
              bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
              partial_fit(x, bin_y)
            end
            # :nocov:
            n_classes.times { |n| @factor_mat[n, true, true], @weight_vec[n, true], @bias_term[n] = models[n] }
          else
            n_classes.times do |n|
              bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
              @factor_mat[n, true, true], @weight_vec[n, true], @bias_term[n] = partial_fit(x, bin_y)
            end
          end
        else
          negative_label = @classes[0]
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @factor_mat, @weight_vec, @bias_term = partial_fit(x, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples]) Confidence score per sample.
      def decision_function(x)
        x = check_convert_sample_array(x)
        linear_term = @bias_term + x.dot(@weight_vec.transpose)
        factor_term = if multiclass_problem?
                        0.5 * (@factor_mat.dot(x.transpose)**2 - (@factor_mat**2).dot(x.transpose**2)).sum(1).transpose
                      else
                        0.5 * (@factor_mat.dot(x.transpose)**2 - (@factor_mat**2).dot(x.transpose**2)).sum(0)
                      end
        linear_term + factor_term
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
        proba = 1.0 / (Numo::NMath.exp(-decision_function(x)) + 1.0)
        return (proba.transpose / proba.sum(axis: 1)).transpose.dup if multiclass_problem?

        n_samples, = x.shape
        probs = Numo::DFloat.zeros(n_samples, 2)
        probs[true, 1] = proba
        probs[true, 0] = 1.0 - proba
        probs
      end

      private

      def bin_decision_function(x, ex_x, factor, weight)
        ex_x.dot(weight) + 0.5 * (factor.dot(x.transpose)**2 - (factor**2).dot(x.transpose**2)).sum(0)
      end

      def loss_func(x, ex_x, y, factor, weight)
        z = bin_decision_function(x, ex_x, factor, weight)
        if @params[:loss] == 'hinge'
          z.class.maximum(0.0, 1 - y * z).sum.fdiv(y.shape[0])
        else
          Numo::NMath.log(1 + Numo::NMath.exp(-y * z)).sum.fdiv(y.shape[0])
        end
      end

      def hinge_loss_gradient(x, ex_x, y, factor, weight)
        evaluated = y * bin_decision_function(x, ex_x, factor, weight)
        gradient = Numo::DFloat.zeros(evaluated.size)
        gradient[evaluated < 1.0] = -y[evaluated < 1.0]
        gradient
      end

      def logistic_loss_gradient(x, ex_x, y, factor, weight)
        evaluated = y * bin_decision_function(x, ex_x, factor, weight)
        sigmoid_func = 1.0 / (Numo::NMath.exp(-evaluated) + 1.0)
        (sigmoid_func - 1.0) * y
      end

      def loss_gradient(x, ex_x, y, factor, weight)
        if @params[:loss] == 'hinge'
          hinge_loss_gradient(x, ex_x, y, factor, weight)
        else
          logistic_loss_gradient(x, ex_x, y, factor, weight)
        end
      end

      def multiclass_problem?
        @classes.size > 2
      end
    end
  end
end
