# frozen_string_literal: true

require 'lbfgsb'

require 'rumale/linear_model/base_sgd'
require 'rumale/base/regressor'

module Rumale
  module LinearModel
    # LinearRegression is a class that implements ordinary least square linear regression
    # with stochastic gradient descent (SGD) optimization,
    # singular value decomposition (SVD), or L-BFGS optimization.
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
    #
    # *Reference*
    # - Bottou, L., "Large-Scale Machine Learning with Stochastic Gradient Descent," Proc. COMPSTAT'10, pp. 177--186, 2010.
    class LinearRegression < BaseSGD
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
      # @param learning_rate [Float] The initial value of learning rate.
      #   The learning rate decreases as the iteration proceeds according to the equation: learning_rate / (1 + decay * t).
      #   If solver is not 'sgd', this parameter is ignored.
      # @param decay [Float] The smoothing parameter for decreasing learning rate as the iteration proceeds.
      #   If nil is given, the decay sets to 'learning_rate'.
      #   If solver is not 'sgd', this parameter is ignored.
      # @param momentum [Float] The momentum factor.
      #   If solver is not 'sgd', this parameter is ignored.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      #   If solver is 'svd', this parameter is ignored.
      # @param batch_size [Integer] The size of the mini batches.
      #   If solver is not 'sgd', this parameter is ignored.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      #   If solver is 'svd', this parameter is ignored.
      # @param solver [String] The algorithm to calculate weights. ('auto', 'sgd', 'svd' or 'lbfgs').
      #   'auto' chooses the 'svd' solver if Numo::Linalg is loaded. Otherwise, it chooses the 'lbfgs' solver.
      #   'sgd' uses the stochastic gradient descent optimization.
      #   'svd' performs singular value decomposition of samples.
      #   'lbfgs' uses the L-BFGS method for optimization.
      # @param n_jobs [Integer] The number of jobs for running the fit method in parallel.
      #   If nil is given, the method does not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded or solver is not 'sgd'.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      #   If solver is 'svd', this parameter is ignored.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(learning_rate: 0.01, decay: nil, momentum: 0.9,
                     fit_bias: true, bias_scale: 1.0, max_iter: 1000, batch_size: 50, tol: 1e-4,
                     solver: 'auto',
                     n_jobs: nil, verbose: false, random_seed: nil)
        check_params_numeric(learning_rate: learning_rate, momentum: momentum,
                             bias_scale: bias_scale, max_iter: max_iter, batch_size: batch_size)
        check_params_boolean(fit_bias: fit_bias, verbose: verbose)
        check_params_string(solver: solver)
        check_params_numeric_or_nil(decay: decay, n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(learning_rate: learning_rate, max_iter: max_iter, batch_size: batch_size)
        super()
        @params.merge!(method(:initialize).parameters.map { |_t, arg| [arg, binding.local_variable_get(arg)] }.to_h)
        @params[:solver] = if solver == 'auto'
                             enable_linalg?(warning: false) ? 'svd' : 'lbfgs'
                           else
                             solver.match?(/^svd$|^sgd$|^lbfgs$/) ? solver : 'lbfgs'
                           end
        @params[:decay] ||= @params[:learning_rate]
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @loss_func = LinearModel::Loss::MeanSquaredError.new
        @weight_vec = nil
        @bias_term = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [LinearRegression] The learned regressor itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_tvalue_array(y)
        check_sample_tvalue_size(x, y)

        if @params[:solver] == 'svd' && enable_linalg?(warning: false)
          fit_svd(x, y)
        elsif @params[:solver] == 'lbfgs'
          fit_lbfgs(x, y)
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
        x = check_convert_sample_array(x)
        x.dot(@weight_vec.transpose) + @bias_term
      end

      private

      def fit_svd(x, y)
        x = expand_feature(x) if fit_bias?
        w = Numo::Linalg.pinv(x, driver: 'svd').dot(y)
        @weight_vec, @bias_term = single_target?(y) ? split_weight(w) : split_weight_mult(w)
      end

      def fit_lbfgs(x, y)
        fnc = proc do |w, x, y| # rubocop:disable Lint/ShadowingOuterLocalVariable
          n_samples, n_features = x.shape
          w = w.reshape(y.shape[1], n_features) unless y.shape[1].nil?
          z = x.dot(w.transpose)
          d = z - y
          loss = (d**2).sum.fdiv(n_samples)
          gradient = 2.fdiv(n_samples) * d.transpose.dot(x)
          [loss, gradient.flatten.dup]
        end

        x = expand_feature(x) if fit_bias?

        n_features = x.shape[1]
        n_outputs = single_target?(y) ? 1 : y.shape[1]

        res = Lbfgsb.minimize(
          fnc: fnc, jcb: true, x_init: init_weight(n_features, n_outputs), args: [x, y],
          maxiter: @params[:max_iter], factr: @params[:tol] / Lbfgsb::DBL_EPSILON,
          verbose: @params[:verbose] ? 1 : -1
        )

        @weight_vec, @bias_term =
          if single_target?(y)
            split_weight(res[:x])
          else
            split_weight_mult(res[:x].reshape(n_outputs, n_features).transpose)
          end
      end

      def fit_sgd(x, y)
        if single_target?(y)
          @weight_vec, @bias_term = partial_fit(x, y)
        else
          n_outputs = y.shape[1]
          n_features = x.shape[1]
          @weight_vec = Numo::DFloat.zeros(n_outputs, n_features)
          @bias_term = Numo::DFloat.zeros(n_outputs)
          if enable_parallel?
            models = parallel_map(n_outputs) { |n| partial_fit(x, y[true, n]) }
            n_outputs.times { |n| @weight_vec[n, true], @bias_term[n] = models[n] }
          else
            n_outputs.times { |n| @weight_vec[n, true], @bias_term[n] = partial_fit(x, y[true, n]) }
          end
        end
      end

      def single_target?(y)
        y.ndim == 1
      end

      def init_weight(n_features, n_outputs)
        Rumale::Utils.rand_normal([n_outputs, n_features], @rng.dup).flatten.dup
      end

      def split_weight_mult(w)
        if fit_bias?
          [w[0...-1, true].dup, w[-1, true].dup]
        else
          [w.dup, Numo::DFloat.zeros(w.shape[1])]
        end
      end
    end
  end
end
