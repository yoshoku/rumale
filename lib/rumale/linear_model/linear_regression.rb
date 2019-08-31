# frozen_string_literal: true

require 'rumale/linear_model/base_linear_model'
require 'rumale/base/regressor'

module Rumale
  module LinearModel
    # LinearRegression is a class that implements ordinary least square linear regression
    # with mini-batch stochastic gradient descent optimization or singular value decomposition.
    #
    # @example
    #   estimator =
    #     Rumale::LinearModel::LinearRegression.new(max_iter: 1000, batch_size: 20, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    #   # If Numo::Linalg is installed, you can specify 'svd' for the solver option.
    #   require 'numo/linalg/autoloader'
    #   estimator = Rumale::LinearModel::LinearRegression.new(solver: 'svd')
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    class LinearRegression < BaseLinearModel
      include Base::Regressor

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_outputs, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept).
      # @return [Numo::DFloat] (shape: [n_outputs])
      attr_reader :bias_term

      # Return the random generator for random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new ordinary least square linear regressor.
      #
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of iterations.
      #   If solver = 'svd', this parameter is ignored.
      # @param batch_size [Integer] The size of the mini batches.
      #   If solver = 'svd', this parameter is ignored.
      # @param optimizer [Optimizer] The optimizer to calculate adaptive learning rate.
      #   If nil is given, Nadam is used.
      #   If solver = 'svd', this parameter is ignored.
      # @param solver [String] The algorithm to calculate weights. ('sgd' or 'svd').
      #   'sgd' uses the stochastic gradient descent optimization.
      #   'svd' performs singular value decomposition of samples.
      # @param n_jobs [Integer] The number of jobs for running the fit method in parallel.
      #   If nil is given, the method does not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(fit_bias: false, bias_scale: 1.0, max_iter: 1000, batch_size: 10, optimizer: nil,
                     solver: 'sgd', n_jobs: nil, random_seed: nil)
        check_params_float(bias_scale: bias_scale)
        check_params_integer(max_iter: max_iter, batch_size: batch_size)
        check_params_boolean(fit_bias: fit_bias)
        check_params_string(solver: solver)
        check_params_type_or_nil(Integer, n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(max_iter: max_iter, batch_size: batch_size)
        keywd_args = method(:initialize).parameters.map { |_t, arg| [arg, binding.local_variable_get(arg)] }.to_h.merge(reg_param: 0.0)
        keywd_args.delete(:solver)
        super(keywd_args)
        @params[:solver] = solver != 'svd' ? 'sgd' : 'svd'
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [LinearRegression] The learned regressor itself.
      def fit(x, y)
        check_sample_array(x)
        check_tvalue_array(y)
        check_sample_tvalue_size(x, y)

        if @params[:solver] == 'svd' && enable_linalg?
          fit_svd(x, y)
        else
          fit_sgd(x, y)
        end

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        check_sample_array(x)
        x.dot(@weight_vec.transpose) + @bias_term
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about LinearRegression.
      def marshal_dump
        { params: @params,
          weight_vec: @weight_vec,
          bias_term: @bias_term,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        @rng = obj[:rng]
        nil
      end

      private

      def fit_svd(x, y)
        samples = @params[:fit_bias] ? expand_feature(x) : x

        s, u, vt = Numo::Linalg.svd(samples, driver: 'sdd', job: 'S')
        d = (s / s**2).diag
        w = vt.transpose.dot(d).dot(u.transpose).dot(y)

        is_single_target_vals = y.shape[1].nil?
        if @params[:fit_bias]
          @weight_vec = is_single_target_vals ? w[0...-1].dup : w[0...-1, true].dup
          @bias_term = is_single_target_vals ? w[-1] : w[-1, true].dup
        else
          @weight_vec = w.dup
          @bias_term = is_single_target_vals ? 0 : Numo::DFloat.zeros(y.shape[1])
        end
      end

      def fit_sgd(x, y)
        n_outputs = y.shape[1].nil? ? 1 : y.shape[1]
        n_features = x.shape[1]

        if n_outputs > 1
          @weight_vec = Numo::DFloat.zeros(n_outputs, n_features)
          @bias_term = Numo::DFloat.zeros(n_outputs)
          if enable_parallel?
            models = parallel_map(n_outputs) { |n| partial_fit(x, y[true, n]) }
            n_outputs.times { |n| @weight_vec[n, true], @bias_term[n] = models[n] }
          else
            n_outputs.times { |n| @weight_vec[n, true], @bias_term[n] = partial_fit(x, y[true, n]) }
          end
        else
          @weight_vec, @bias_term = partial_fit(x, y)
        end
      end

      def calc_loss_gradient(x, y, weight)
        2.0 * (x.dot(weight) - y)
      end
    end
  end
end
