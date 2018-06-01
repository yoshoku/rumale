# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'
require 'svmkit/optimizer/nadam'

module SVMKit
  # This module consists of the classes that implement polynomial models.
  module PolynomialModel
    # FactorizationMachineClassifier is a class that implements Factorization Machine
    # with stochastic gradient descent (SGD) optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # @example
    #   estimator =
    #     SVMKit::PolynomialModel::FactorizationMachineClassifier.new(
    #      n_factors: 10, loss: 'hinge', reg_param_linear: 0.001, reg_param_factor: 0.001,
    #      max_iter: 5000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Rendle, "Factorization Machines with libFM," ACM Transactions on Intelligent Systems and Technology, vol. 3 (3), pp. 57:1--57:22, 2012.
    # - S. Rendle, "Factorization Machines," Proceedings of the 10th IEEE International Conference on Data Mining (ICDM'10), pp. 995--1000, 2010.
    class FactorizationMachineClassifier
      include Base::BaseEstimator
      include Base::Classifier
      include Validation

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
      #   Nadam is selected automatically on current version.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2, loss: 'hinge', reg_param_linear: 1.0, reg_param_factor: 1.0,
                     max_iter: 1000, batch_size: 10, optimizer: nil, random_seed: nil)
        check_params_float(reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor)
        check_params_integer(n_factors: n_factors, max_iter: max_iter, batch_size: batch_size)
        check_params_string(loss: loss)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        check_params_positive(n_factors: n_factors,
                              reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor,
                              max_iter: max_iter, batch_size: batch_size)
        @params = {}
        @params[:n_factors] = n_factors
        @params[:loss] = loss
        @params[:reg_param_linear] = reg_param_linear
        @params[:reg_param_factor] = reg_param_factor
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:optimizer] = optimizer
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @factor_mat = nil
        @weight_vec = nil
        @bias_term = nil
        @classes = nil
        @rng = Random.new(@params[:random_seed])
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
          n_classes.times do |n|
            bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
            factor, weight, bias = binary_fit(x, bin_y)
            @factor_mat[n, true, true] = factor
            @weight_vec[n, true] = weight
            @bias_term[n] = bias
          end
        else
          negative_label = y.to_a.uniq.min
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @factor_mat, @weight_vec, @bias_term = binary_fit(x, bin_y)
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

        n_samples, = x.shape
        decision_values = decision_function(x)
        Numo::Int32.asarray(Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] })
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

      def binary_fit(x, y)
        # Initialize some variables.
        n_samples, n_features = x.shape
        rand_ids = [*0...n_samples].shuffle(random: @rng)
        weight_vec = Numo::DFloat.zeros(n_features + 1)
        factor_mat = Numo::DFloat.zeros(@params[:n_factors], n_features)
        weight_optimizer = Optimizer::Nadam.new
        factor_optimizers = Array.new(@params[:n_factors]) { Optimizer::Nadam.new }
        # Start optimization.
        @params[:max_iter].times do |_t|
          # Random sampling.
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          data = x[subset_ids, true]
          ex_data = expand_feature(data)
          label = y[subset_ids]
          # Calculate gradients for loss function.
          loss_grad = loss_gradient(data, ex_data, label, factor_mat, weight_vec)
          next if loss_grad.ne(0.0).count.zero?
          # Update each parameter.
          weight_vec = weight_optimizer.call(weight_vec, weight_gradient(loss_grad, ex_data, weight_vec))
          @params[:n_factors].times do |n|
            factor_mat[n, true] = factor_optimizers[n].call(factor_mat[n, true],
                                                            factor_gradient(loss_grad, data, factor_mat[n, true]))
          end
        end
        [factor_mat, *split_weight_vec_bias(weight_vec)]
      end

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

      def weight_gradient(loss_grad, data, weight)
        (loss_grad.expand_dims(1) * data).mean(0) + @params[:reg_param_linear] * weight
      end

      def factor_gradient(loss_grad, data, factor)
        (loss_grad.expand_dims(1) * (data * data.dot(factor).expand_dims(1) - factor * (data**2))).mean(0) + @params[:reg_param_factor] * factor
      end

      def expand_feature(x)
        Numo::NArray.hstack([x, Numo::DFloat.ones([x.shape[0], 1])])
      end

      def split_weight_vec_bias(weight_vec)
        weights = weight_vec[0...-1]
        bias = weight_vec[-1]
        [weights, bias]
      end
    end
  end
end
