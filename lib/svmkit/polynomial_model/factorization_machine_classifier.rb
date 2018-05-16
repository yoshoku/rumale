# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

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
    #      n_factors: 10, loss: 'hinge', reg_param_bias: 0.001, reg_param_weight: 0.001, reg_param_factor: 0.001,
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
      # @param reg_param_bias [Float] The regularization parameter for bias term.
      # @param reg_param_weight [Float] The regularization parameter for weight vector.
      # @param reg_param_factor [Float] The regularization parameter for factor matrix.
      # @param init_std [Float] The standard deviation of normal random number for initialization of factor matrix.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2, loss: 'hinge', reg_param_bias: 1.0, reg_param_weight: 1.0, reg_param_factor: 1.0,
                     init_std: 0.1, max_iter: 1000, batch_size: 10, random_seed: nil)
        SVMKit::Validation.check_params_float(reg_param_bias: reg_param_bias, reg_param_weight: reg_param_weight,
                                              reg_param_factor: reg_param_factor, init_std: init_std)
        SVMKit::Validation.check_params_integer(n_factors: n_factors, max_iter: max_iter, batch_size: batch_size)
        SVMKit::Validation.check_params_string(loss: loss)
        SVMKit::Validation.check_params_type_or_nil(Integer, random_seed: random_seed)
        SVMKit::Validation.check_params_positive(n_factors: n_factors, reg_param_bias: reg_param_bias,
                                                 reg_param_weight: reg_param_weight, reg_param_factor: reg_param_factor,
                                                 max_iter: max_iter, batch_size: batch_size)
        @params = {}
        @params[:n_factors] = n_factors
        @params[:loss] = loss
        @params[:reg_param_bias] = reg_param_bias
        @params[:reg_param_weight] = reg_param_weight
        @params[:reg_param_factor] = reg_param_factor
        @params[:init_std] = init_std
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
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
        SVMKit::Validation.check_sample_array(x)
        SVMKit::Validation.check_label_array(y)
        SVMKit::Validation.check_sample_label_size(x, y)

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
        SVMKit::Validation.check_sample_array(x)
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
        SVMKit::Validation.check_sample_array(x)
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
        SVMKit::Validation.check_sample_array(x)
        proba = 1.0 / (Numo::NMath.exp(-decision_function(x)) + 1.0)
        return (proba.transpose / proba.sum(axis: 1)).transpose if @classes.size > 2

        n_samples, = x.shape
        probs = Numo::DFloat.zeros(n_samples, 2)
        probs[true, 1] = proba
        probs[true, 0] = 1.0 - proba
        probs
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about FactorizationMachineClassifier
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

      def binary_fit(x, bin_y)
        # Initialize some variables.
        n_samples, n_features = x.shape
        rand_ids = [*0...n_samples].shuffle(random: @rng)
        factor_mat = rand_normal([@params[:n_factors], n_features], 0, @params[:init_std])
        weight_vec = Numo::DFloat.zeros(n_features)
        bias_term = 0.0
        # Start optimization.
        @params[:max_iter].times do |t|
          # Random sampling.
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          data = x[subset_ids, true]
          label = bin_y[subset_ids]
          # Calculate gradients for loss function.
          loss_grad = loss_gradient(data, label, factor_mat, weight_vec, bias_term)
          next if loss_grad.ne(0.0).count.zero?
          # Update each parameter.
          bias_term -= learning_rate(@params[:reg_param_bias], t) * bias_gradient(loss_grad, bias_term)
          weight_vec -= learning_rate(@params[:reg_param_weight], t) * weight_gradient(loss_grad, data, weight_vec)
          @params[:n_factors].times do |n|
            factor_mat[n, true] -= learning_rate(@params[:reg_param_factor], t) *
                                   factor_gradient(loss_grad, data, factor_mat[n, true])
          end
        end
        [factor_mat, weight_vec, bias_term]
      end

      def bin_decision_function(x, factor, weight, bias)
        bias + x.dot(weight) + 0.5 * (factor.dot(x.transpose)**2 - (factor**2).dot(x.transpose**2)).sum(0)
      end

      def hinge_loss_gradient(x, y, factor, weight, bias)
        evaluated = y * bin_decision_function(x, factor, weight, bias)
        gradient = Numo::DFloat.zeros(evaluated.size)
        gradient[evaluated < 1.0] = -y[evaluated < 1.0]
        gradient
      end

      def logistic_loss_gradient(x, y, factor, weight, bias)
        evaluated = y * bin_decision_function(x, factor, weight, bias)
        sigmoid_func = 1.0 / (Numo::NMath.exp(-evaluated) + 1.0)
        (sigmoid_func - 1.0) * y
      end

      def loss_gradient(x, y, factor, weight, bias)
        if @params[:loss] == 'hinge'
          hinge_loss_gradient(x, y, factor, weight, bias)
        else
          logistic_loss_gradient(x, y, factor, weight, bias)
        end
      end

      def learning_rate(reg_param, iter)
        1.0 / (reg_param * (iter + 1))
      end

      def bias_gradient(loss_grad, bias)
        loss_grad.mean + @params[:reg_param_bias] * bias
      end

      def weight_gradient(loss_grad, data, weight)
        (loss_grad.expand_dims(1) * data).mean(0) + @params[:reg_param_weight] * weight
      end

      def factor_gradient(loss_grad, data, factor)
        reg_term = @params[:reg_param_factor] * factor
        (loss_grad.expand_dims(1) * (data * data.dot(factor).expand_dims(1) - factor * (data**2))).mean(0) + reg_term
      end

      def rand_uniform(shape)
        Numo::DFloat[*Array.new(shape.inject(&:*)) { @rng.rand }].reshape(*shape)
      end

      def rand_normal(shape, mu, sigma)
        a = rand_uniform(shape)
        b = rand_uniform(shape)
        mu + sigma * (Numo::NMath.sqrt(-2.0 * Numo::NMath.log(a)) * Numo::NMath.sin(2.0 * Math::PI * b))
      end
    end
  end
end
