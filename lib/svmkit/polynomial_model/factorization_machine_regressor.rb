# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/regressor'
require 'svmkit/optimizer/nadam'

module SVMKit
  module PolynomialModel
    # FactorizationMachineRegressor is a class that implements Factorization Machine
    # with stochastic gradient descent (SGD) optimization.
    #
    # @example
    #   estimator =
    #     SVMKit::PolynomialModel::FactorizationMachineRegressor.new(
    #      n_factors: 10, reg_param_linear: 0.1, reg_param_factor: 0.1,
    #      max_iter: 5000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Rendle, "Factorization Machines with libFM," ACM TIST, vol. 3 (3), pp. 57:1--57:22, 2012.
    # - S. Rendle, "Factorization Machines," Proc. ICDM'10, pp. 995--1000, 2010.
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
      # @param reg_param_linear [Float] The regularization parameter for linear model.
      # @param reg_param_factor [Float] The regularization parameter for factor matrix.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param optimizer [Optimizer] The optimizer to calculate adaptive learning rate.
      #   If nil is given, Nadam is used.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2, reg_param_linear: 1.0, reg_param_factor: 1.0,
                     max_iter: 1000, batch_size: 10, optimizer: nil, random_seed: nil)
        check_params_float(reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor)
        check_params_integer(n_factors: n_factors, max_iter: max_iter, batch_size: batch_size)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        check_params_positive(n_factors: n_factors, reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor,
                              max_iter: max_iter, batch_size: batch_size)
        @params = {}
        @params[:n_factors] = n_factors
        @params[:reg_param_linear] = reg_param_linear
        @params[:reg_param_factor] = reg_param_factor
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:optimizer] = optimizer
        @params[:optimizer] ||= Optimizer::Nadam.new
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
          n_outputs.times { |n| @factor_mat[n, true, true], @weight_vec[n, true], @bias_term[n] = single_fit(x, y[true, n]) }
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
      # @return [Hash] The marshal data about FactorizationMachineRegressor.
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
        weight_vec = Numo::DFloat.zeros(n_features + 1)
        factor_mat = Numo::DFloat.zeros(@params[:n_factors], n_features)
        weight_optimizer = @params[:optimizer].dup
        factor_optimizers = Array.new(@params[:n_factors]) { @params[:optimizer].dup }
        # Start optimization.
        @params[:max_iter].times do |_t|
          # Random sampling.
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          data = x[subset_ids, true]
          ex_data = expand_feature(data)
          values = y[subset_ids]
          # Calculate gradients for loss function.
          loss_grad = loss_gradient(data, ex_data, values, factor_mat, weight_vec)
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

      def loss_gradient(x, ex_x, y, factor, weight)
        z = ex_x.dot(weight) + 0.5 * (factor.dot(x.transpose)**2 - (factor**2).dot(x.transpose**2)).sum(0)
        2.0 * (z - y)
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
