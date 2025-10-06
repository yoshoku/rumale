# frozen_string_literal: true

require 'numo/optimize'

require 'rumale/base/regressor'
require 'rumale/validation'

require_relative 'base_estimator'

module Rumale
  module LinearModel
    # Ridge is a class that implements Ridge Regression
    # with singular value decomposition (SVD) or L-BFGS optimization.
    #
    # @example
    #   require 'rumale/linear_model/ridge'
    #
    #   estimator = Rumale::LinearModel::Ridge.new(reg_param: 0.1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    #   # If Numo::Linalg is installed, you can specify 'svd' for the solver option.
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/linear_model/ridge'
    #
    #   estimator = Rumale::LinearModel::Ridge.new(reg_param: 0.1, solver: 'svd')
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    class Ridge < Rumale::LinearModel::BaseEstimator
      include Rumale::Base::Regressor

      # Create a new Ridge regressor.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      #   If solver is 'svd', this parameter is ignored.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      #   If solver is 'svd', this parameter is ignored.
      # @param solver [String] The algorithm to calculate weights. ('auto', 'svd', or 'lbfgs').
      #   'auto' chooses the 'svd' solver if Numo::Linalg is loaded. Otherwise, it chooses the 'lbfgs' solver.
      #   'svd' performs singular value decomposition of samples.
      #   'lbfgs' uses the L-BFGS method for optimization.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      #   If solver is 'svd', this parameter is ignored.
      def initialize(reg_param: 1.0, fit_bias: true, bias_scale: 1.0, max_iter: 1000, tol: 1e-4, solver: 'auto', verbose: false)
        super()
        @params = {
          reg_param: reg_param,
          fit_bias: fit_bias,
          bias_scale: bias_scale,
          max_iter: max_iter,
          tol: tol,
          verbose: verbose
        }
        @params[:solver] = if solver == 'auto'
                             enable_linalg?(warning: false) ? 'svd' : 'lbfgs'
                           else
                             solver.match?(/^svd$|^lbfgs$/) ? solver : 'lbfgs'
                           end
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [Ridge] The learned regressor itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_target_value_array(y)
        Rumale::Validation.check_sample_size(x, y)

        @weight_vec, @bias_term = if @params[:solver] == 'svd' && enable_linalg?(warning: false)
                                    partial_fit_svd(x, y)
                                  else
                                    partial_fit_lbfgs(x, y)
                                  end

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        x = Rumale::Validation.check_convert_sample_array(x)

        x.dot(@weight_vec.transpose) + @bias_term
      end

      private

      def partial_fit_svd(x, y)
        x = expand_feature(x) if fit_bias?
        s, u, vt = Numo::Linalg.svd(x, driver: 'sdd', job: 'S')
        d = (s / (s**2 + @params[:reg_param])).diag
        w = vt.transpose.dot(d).dot(u.transpose).dot(y)
        w = w.transpose.dup unless single_target?(y)
        split_weight(w)
      end

      def partial_fit_lbfgs(base_x, base_y)
        fnc = proc do |w, x, y, a|
          n_samples, n_features = x.shape
          w = w.reshape(y.shape[1], n_features) unless y.shape[1].nil?
          z = x.dot(w.transpose)
          d = z - y
          loss = (d**2).sum.fdiv(n_samples) + a * (w * w).sum
          gradient = 2.fdiv(n_samples) * d.transpose.dot(x) + 2.0 * a * w
          [loss, gradient.flatten.dup]
        end

        base_x = expand_feature(base_x) if fit_bias?

        n_features = base_x.shape[1]
        n_outputs = single_target?(base_y) ? 1 : base_y.shape[1]
        w_init = Numo::DFloat.zeros(n_outputs * n_features)

        res = Numo::Optimize.minimize(
          fnc: fnc, jcb: true, x_init: w_init, args: [base_x, base_y, @params[:reg_param]],
          maxiter: @params[:max_iter], factr: @params[:tol] / Numo::Optimize::Lbfgsb::DBL_EPSILON,
          verbose: @params[:verbose] ? 1 : -1
        )

        w = single_target?(base_y) ? res[:x] : res[:x].reshape(n_outputs, n_features)
        split_weight(w)
      end

      def single_target?(y)
        y.ndim == 1
      end
    end
  end
end
