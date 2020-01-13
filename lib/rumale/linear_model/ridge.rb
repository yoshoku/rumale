# frozen_string_literal: true

require 'rumale/linear_model/base_sgd'
require 'rumale/base/regressor'

module Rumale
  module LinearModel
    # Ridge is a class that implements Ridge Regression
    # with stochastic gradient descent (SGD) optimization or singular value decomposition (SVD).
    #
    # @example
    #   estimator =
    #     Rumale::LinearModel::Ridge.new(reg_param: 0.1, max_iter: 500, batch_size: 20, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    #   # If Numo::Linalg is installed, you can specify 'svd' for the solver option.
    #   require 'numo/linalg/autoloader'
    #   estimator = Rumale::LinearModel::Ridge.new(reg_param: 0.1, solver: 'svd')
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - L. Bottou, "Large-Scale Machine Learning with Stochastic Gradient Descent," Proc. COMPSTAT'10, pp. 177--186, 2010.
    class Ridge < BaseSGD
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

      # Create a new Ridge regressor.
      #
      # @param learning_rate [Float] The initial value of learning rate.
      #   The learning rate decreases as the iteration proceeds according to the equation: learning_rate / (1 + decay * t).
      #   If solver = 'svd', this parameter is ignored.
      # @param decay [Float] The smoothing parameter for decreasing learning rate as the iteration proceeds.
      #   If nil is given, the decay sets to 'reg_param * learning_rate'.
      #   If solver = 'svd', this parameter is ignored.
      # @param momentum [Float] The momentum factor.
      #   If solver = 'svd', this parameter is ignored.
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      #   If solver = 'svd', this parameter is ignored.
      # @param batch_size [Integer] The size of the mini batches.
      #   If solver = 'svd', this parameter is ignored.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      #   If solver = 'svd', this parameter is ignored.
      # @param solver [String] The algorithm to calculate weights. ('auto', 'sgd' or 'svd').
      #   'auto' chooses the 'svd' solver if Numo::Linalg is loaded. Otherwise, it chooses the 'sgd' solver.
      #   'sgd' uses the stochastic gradient descent optimization.
      #   'svd' performs singular value decomposition of samples.
      # @param n_jobs [Integer] The number of jobs for running the fit method in parallel.
      #   If nil is given, the method does not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded or the solver is 'svd'.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      #   If solver = 'svd', this parameter is ignored.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(learning_rate: 0.01, decay: nil, momentum: 0.9,
                     reg_param: 1.0, fit_bias: true, bias_scale: 1.0,
                     max_iter: 100, batch_size: 50, tol: 1e-4,
                     solver: 'auto',
                     n_jobs: nil, verbose: false, random_seed: nil)
        check_params_numeric(learning_rate: learning_rate, momentum: momentum,
                             reg_param: reg_param, bias_scale: bias_scale,
                             max_iter: max_iter, batch_size: batch_size, tol: tol)
        check_params_boolean(fit_bias: fit_bias, verbose: verbose)
        check_params_string(solver: solver)
        check_params_numeric_or_nil(decay: decay, n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(learning_rate: learning_rate, reg_param: reg_param, max_iter: max_iter, batch_size: batch_size)
        super()
        @params.merge!(method(:initialize).parameters.map { |_t, arg| [arg, binding.local_variable_get(arg)] }.to_h)
        @params[:solver] = if solver == 'auto'
                             load_linalg? ? 'svd' : 'sgd'
                           else
                             solver != 'svd' ? 'sgd' : 'svd'
                           end
        @params[:decay] ||= @params[:reg_param] * @params[:learning_rate]
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @penalty_type = L2_PENALTY
        @loss_func = LinearModel::Loss::MeanSquaredError.new
        @weight_vec = nil
        @bias_term = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [Ridge] The learned regressor itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_tvalue_array(y)
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
        x = check_convert_sample_array(x)
        x.dot(@weight_vec.transpose) + @bias_term
      end

      private

      def fit_svd(x, y)
        samples = @params[:fit_bias] ? expand_feature(x) : x

        s, u, vt = Numo::Linalg.svd(samples, driver: 'sdd', job: 'S')
        d = (s / (s**2 + @params[:reg_param])).diag
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

      def load_linalg?
        return false if defined?(Numo::Linalg).nil?
        return false if Numo::Linalg::VERSION < '0.1.4'
        true
      end
    end
  end
end
