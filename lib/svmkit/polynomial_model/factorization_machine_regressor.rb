# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/regressor'

module SVMKit
  module PolynomialModel
    # FactorizationMachineRegressor is a class that implements Factorization Machine
    # with stochastic gradient descent (SGD) optimization.
    #
    # @example
    #   estimator =
    #     SVMKit::PolynomialModel::FactorizationMachineRegressor.new(
    #      n_factors: 10, reg_param_bias: 0.1, reg_param_weight: 0.1, reg_param_factor: 0.1,
    #      max_iter: 5000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Rendle, "Factorization Machines with libFM," ACM Transactions on Intelligent Systems and Technology, vol. 3 (3), pp. 57:1--57:22, 2012.
    # - S. Rendle, "Factorization Machines," Proc. the 10th IEEE International Conference on Data Mining (ICDM'10), pp. 995--1000, 2010.
    # - I. Sutskever, J. Martens, G. Dahl, and G. Hinton, "On the importance of initialization and momentum in deep learning," Proc. the 30th  International Conference on Machine Learning (ICML' 13), pp. 1139--1147, 2013.
    # - G. Hinton, N. Srivastava, and K. Swersky, "Lecture 6e rmsprop," Neural Networks for Machine Learning, 2012.
    class FactorizationMachineRegressor
      include Base::BaseEstimator
      include Base::Regressor
      include Validation

      # Return the factor matrix for Factorization Machine.
      # @return [Numo::DFloat] (shape: [n_outputs, n_factors, n_features])
      attr_reader :factor_mat

      # Return the weight vector for Factorization Machine.
      # @return [Numo::DFloat] (shape: [n_outputs, n_features])
      attr_reader :weight_vec

      # Return the bias term for Factoriazation Machine.
      # @return [Numo::DFloat] (shape: [n_outputs])
      attr_reader :bias_term

      # Return the random generator for random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with Factorization Machine.
      #
      # @param n_factors [Integer] The maximum number of iterations.
      # @param reg_param_bias [Float] The regularization parameter for bias term.
      # @param reg_param_weight [Float] The regularization parameter for weight vector.
      # @param reg_param_factor [Float] The regularization parameter for factor matrix.
      # @param init_std [Float] The standard deviation of normal random number for initialization of factor matrix.
      # @param learning_rate [Float] The learning rate for optimization.
      # @param decay [Float] The discounting factor for RMS prop optimization.
      # @param momentum [Float] The Nesterov momentum for optimization.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2,
                     reg_param_bias: 1.0, reg_param_weight: 1.0, reg_param_factor: 1.0, init_std: 0.01,
                     learning_rate: 0.01, decay: 0.9, momentum: 0.9,
                     max_iter: 1000, batch_size: 10, random_seed: nil)
        check_params_float(reg_param_bias: reg_param_bias, reg_param_weight: reg_param_weight,
                           reg_param_factor: reg_param_factor, init_std: init_std,
                           learning_rate: learning_rate, decay: decay, momentum: momentum)
        check_params_integer(n_factors: n_factors, max_iter: max_iter, batch_size: batch_size)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        check_params_positive(n_factors: n_factors, reg_param_bias: reg_param_bias,
                              reg_param_weight: reg_param_weight, reg_param_factor: reg_param_factor,
                              learning_rate: learning_rate, decay: decay, momentum: momentum,
                              max_iter: max_iter, batch_size: batch_size)
        @params = {}
        @params[:n_factors] = n_factors
        @params[:reg_param_bias] = reg_param_bias
        @params[:reg_param_weight] = reg_param_weight
        @params[:reg_param_factor] = reg_param_factor
        @params[:init_std] = init_std
        @params[:learning_rate] = learning_rate
        @params[:decay] = decay
        @params[:momentum] = momentum
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @factor_mat = nil
        @weight_vec = nil
        @bias_term = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [FactorizationMachineRegressor] The learned regressor itself.
      def fit(x, y)
        check_sample_array(x)
        check_tvalue_array(y)
        check_sample_tvalue_size(x, y)

        n_outputs = y.shape[1].nil? ? 1 : y.shape[1]
        _n_samples, n_features = x.shape

        if n_outputs > 1
          @factor_mat = Numo::DFloat.zeros(n_outputs, @params[:n_factors], n_features)
          @weight_vec = Numo::DFloat.zeros(n_outputs, n_features)
          @bias_term = Numo::DFloat.zeros(n_outputs)
          n_outputs.times do |n|
            factor, weight, bias = single_fit(x, y[true, n])
            @factor_mat[n, true, true] = factor
            @weight_vec[n, true] = weight
            @bias_term[n] = bias
          end
        else
          @factor_mat, @weight_vec, @bias_term = single_fit(x, y)
        end

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        check_sample_array(x)
        linear_term = @bias_term + x.dot(@weight_vec.transpose)
        factor_term = if @weight_vec.shape[1].nil?
                        0.5 * (@factor_mat.dot(x.transpose)**2 - (@factor_mat**2).dot(x.transpose**2)).sum(0)
                      else
                        0.5 * (@factor_mat.dot(x.transpose)**2 - (@factor_mat**2).dot(x.transpose**2)).sum(1).transpose
                      end
        linear_term + factor_term
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about FactorizationMachineRegressor
      def marshal_dump
        { params: @params,
          factor_mat: @factor_mat,
          weight_vec: @weight_vec,
          bias_term: @bias_term,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @factor_mat = obj[:factor_mat]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        @rng = obj[:rng]
        nil
      end

      private

      def single_fit(x, y)
        # Initialize some variables.
        n_samples, n_features = x.shape
        rand_ids = [*0...n_samples].shuffle(random: @rng)
        factor_mat = rand_normal([@params[:n_factors], n_features], 0, @params[:init_std])
        factor_sqrsum = Numo::DFloat.zeros(factor_mat.shape)
        factor_update = Numo::DFloat.zeros(factor_mat.shape)
        weight_vec = Numo::DFloat.zeros(n_features)
        weight_sqrsum = Numo::DFloat.zeros(n_features)
        weight_update = Numo::DFloat.zeros(n_features)
        bias_term = 0.0
        bias_sqrsum = 0.0
        bias_update = 0.0
        # Start optimization.
        @params[:max_iter].times do |_t|
          # Random sampling.
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          data = x[subset_ids, true]
          values = y[subset_ids]
          # Calculate gradients for loss function.
          loss_grad = loss_gradient(data, values, factor_mat, weight_vec, bias_term)
          next if loss_grad.ne(0.0).count.zero?
          # Update each parameter.
          bias_term, bias_sqrsum, bias_update =
            update_param(bias_term, bias_sqrsum, bias_update,
                         bias_gradient(loss_grad, bias_term - @params[:momentum] * bias_update))
          weight_vec, weight_sqrsum, weight_update =
            update_param(weight_vec, weight_sqrsum, weight_update,
                         weight_gradient(loss_grad, data, weight_vec - @params[:momentum] * weight_update))
          @params[:n_factors].times do |n|
            factor_update[n, true], factor_sqrsum[n, true], factor_update[n, true] =
              update_param(factor_update[n, true], factor_sqrsum[n, true], factor_update[n, true],
                           factor_gradient(loss_grad, data, factor_mat[n, true] - @params[:momentum] * factor_update[n, true]))
          end
        end
        [factor_mat, weight_vec, bias_term]
      end

      def loss_gradient(x, y, factor, weight, bias)
        z = bias + x.dot(weight) + 0.5 * (factor.dot(x.transpose)**2 - (factor**2).dot(x.transpose**2)).sum(0)
        2.0 * (z - y)
      end

      def bias_gradient(loss_grad, bias)
        loss_grad.mean + @params[:reg_param_bias] * bias
      end

      def weight_gradient(loss_grad, data, weight)
        (loss_grad.expand_dims(1) * data).mean(0) + @params[:reg_param_weight] * weight
      end

      def factor_gradient(loss_grad, data, factor)
        (loss_grad.expand_dims(1) * (data * data.dot(factor).expand_dims(1) - factor * (data**2))).mean(0) + @params[:reg_param_factor] * factor
      end

      def update_param(param, sqrsum, update, gr)
        new_sqrsum = @params[:decay] * sqrsum + (1.0 - @params[:decay]) * gr**2
        new_update = (@params[:learning_rate] / ((new_sqrsum + 1.0e-8)**0.5)) * gr
        new_param = param - (new_update + @params[:momentum] * update)
        [new_param, new_sqrsum, new_update]
      end

      def rand_uniform(shape)
        Numo::DFloat[*Array.new(shape.inject(&:*)) { @rng.rand }].reshape(*shape)
      end

      def rand_normal(shape, mu, sigma)
        mu + sigma * (Numo::NMath.sqrt(-2.0 * Numo::NMath.log(rand_uniform(shape))) * Numo::NMath.sin(2.0 * Math::PI * rand_uniform(shape)))
      end
    end
  end
end
