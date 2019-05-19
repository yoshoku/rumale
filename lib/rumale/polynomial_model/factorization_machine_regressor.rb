# frozen_string_literal: true

require 'rumale/base/regressor'
require 'rumale/polynomial_model/base_factorization_machine'

module Rumale
  module PolynomialModel
    # FactorizationMachineRegressor is a class that implements Factorization Machine
    # with stochastic gradient descent (SGD) optimization.
    #
    # @example
    #   estimator =
    #     Rumale::PolynomialModel::FactorizationMachineRegressor.new(
    #      n_factors: 10, reg_param_linear: 0.1, reg_param_factor: 0.1,
    #      max_iter: 5000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Rendle, "Factorization Machines with libFM," ACM TIST, vol. 3 (3), pp. 57:1--57:22, 2012.
    # - S. Rendle, "Factorization Machines," Proc. ICDM'10, pp. 995--1000, 2010.
    class FactorizationMachineRegressor < BaseFactorizationMachine
      include Base::Regressor

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
      # @param n_jobs [Integer] The number of jobs for running the fit method in parallel.
      #   If nil is given, the method does not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2, reg_param_linear: 1.0, reg_param_factor: 1.0,
                     max_iter: 1000, batch_size: 10, optimizer: nil, n_jobs: nil, random_seed: nil)
        check_params_float(reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor)
        check_params_integer(n_factors: n_factors, max_iter: max_iter, batch_size: batch_size)
        check_params_type_or_nil(Integer, n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(n_factors: n_factors, reg_param_linear: reg_param_linear, reg_param_factor: reg_param_factor,
                              max_iter: max_iter, batch_size: batch_size)
        keywd_args = method(:initialize).parameters.map { |_t, arg| [arg, binding.local_variable_get(arg)] }.to_h.merge(loss: nil)
        super(keywd_args)
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
          if enable_parallel?
            models = parallel_map(n_outputs) { |n| partial_fit(x, y[true, n]) }
            n_outputs.times { |n| @factor_mat[n, true, true], @weight_vec[n, true], @bias_term[n] = models[n] }
          else
            n_outputs.times { |n| @factor_mat[n, true, true], @weight_vec[n, true], @bias_term[n] = partial_fit(x, y[true, n]) }
          end
        else
          @factor_mat, @weight_vec, @bias_term = partial_fit(x, y)
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

      def loss_gradient(x, ex_x, y, factor, weight)
        z = ex_x.dot(weight) + 0.5 * (factor.dot(x.transpose)**2 - (factor**2).dot(x.transpose**2)).sum(0)
        2.0 * (z - y)
      end
    end
  end
end
