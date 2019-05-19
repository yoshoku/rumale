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
    #      max_iter: 5000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Rendle, "Factorization Machines with libFM," ACM TIST, vol. 3 (3), pp. 57:1--57:22, 2012.
    # - S. Rendle, "Factorization Machines," Proc. ICDM'10, pp. 995--1000, 2010.
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
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param optimizer [Optimizer] The optimizer to calculate adaptive learning rate.
      #   If nil is given, Nadam is used.
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2, loss: 'hinge', reg_param_linear: 1.0, reg_param_factor: 1.0,
                     max_iter: 1000, batch_size: 10, optimizer: nil, n_jobs: nil, random_seed: nil)
        check_params_float(reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor)
        check_params_integer(n_factors: n_factors, max_iter: max_iter, batch_size: batch_size)
        check_params_string(loss: loss)
        check_params_type_or_nil(Integer, n_jobs: n_jobs, random_seed: random_seed)
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
        check_sample_array(x)
        check_label_array(y)
        check_sample_label_size(x, y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_classes = @classes.size
        _n_samples, n_features = x.shape

        if n_classes > 2
          @factor_mat = Numo::DFloat.zeros(n_classes, @params[:n_factors], n_features)
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          if enable_parallel?
            models = parallel_map(n_classes) do |n|
              bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
              partial_fit(x, bin_y)
            end
            n_classes.times { |n| @factor_mat[n, true, true], @weight_vec[n, true], @bias_term[n] = models[n] }
          else
            n_classes.times do |n|
              bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
              @factor_mat[n, true, true], @weight_vec[n, true], @bias_term[n] = partial_fit(x, bin_y)
            end
          end
        else
          negative_label = y.to_a.uniq.min
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
        check_sample_array(x)
        linear_term = @bias_term + x.dot(@weight_vec.transpose)
        factor_term = if @classes.size <= 2
                        0.5 * (@factor_mat.dot(x.transpose)**2 - (@factor_mat**2).dot(x.transpose**2)).sum(0)
                      else
                        0.5 * (@factor_mat.dot(x.transpose)**2 - (@factor_mat**2).dot(x.transpose**2)).sum(1).transpose
                      end
        linear_term + factor_term
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        check_sample_array(x)
        return Numo::Int32.cast(decision_function(x).ge(0.0)) * 2 - 1 if @classes.size <= 2

        n_samples = x.shape[0]
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
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        check_sample_array(x)
        proba = 1.0 / (Numo::NMath.exp(-decision_function(x)) + 1.0)
        return (proba.transpose / proba.sum(axis: 1)).transpose if @classes.size > 2

        n_samples, = x.shape
        probs = Numo::DFloat.zeros(n_samples, 2)
        probs[true, 1] = proba
        probs[true, 0] = 1.0 - proba
        probs
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about FactorizationMachineClassifier.
      def marshal_dump
        { params: @params,
          factor_mat: @factor_mat,
          weight_vec: @weight_vec,
          bias_term: @bias_term,
          classes: @classes,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @factor_mat = obj[:factor_mat]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        @classes = obj[:classes]
        @rng = obj[:rng]
        nil
      end

      private

      def bin_decision_function(x, ex_x, factor, weight)
        ex_x.dot(weight) + 0.5 * (factor.dot(x.transpose)**2 - (factor**2).dot(x.transpose**2)).sum(0)
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
    end
  end
end
