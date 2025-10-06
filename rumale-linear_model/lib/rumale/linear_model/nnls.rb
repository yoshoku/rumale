# frozen_string_literal: true

require 'numo/optimize'

require 'rumale/base/regressor'
require 'rumale/validation'

require_relative 'base_estimator'

module Rumale
  module LinearModel
    # NNLS is a class that implements non-negative least squares regression.
    # NNLS solves least squares problem under non-negative constraints on the coefficient using L-BFGS-B method.
    #
    # @example
    #   require 'rumale/linear_model/nnls'
    #
    #   estimator = Rumale::LinearModel::NNLS.new(reg_param: 0.01)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    class NNLS < Rumale::LinearModel::BaseEstimator
      include Rumale::Base::Regressor

      # Returns the number of iterations when converged.
      # @return [Integer]
      attr_reader :n_iter

      # Create a new regressor with non-negative least squares method.
      #
      # @param reg_param [Float] The regularization parameter for L2 regularization term.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      #   If solver = 'svd', this parameter is ignored.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      def initialize(reg_param: 1.0, fit_bias: true, bias_scale: 1.0, max_iter: 1000, tol: 1e-4, verbose: false)
        super()
        @params = {
          reg_param: reg_param,
          fit_bias: fit_bias,
          bias_scale: bias_scale,
          max_iter: max_iter,
          tol: tol,
          verbose: verbose
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [NonneagtiveLeastSquare] The learned regressor itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_target_value_array(y)
        Rumale::Validation.check_sample_size(x, y)

        x = expand_feature(x) if fit_bias?

        n_features = x.shape[1]
        n_outputs = single_target?(y) ? 1 : y.shape[1]

        w_init = Numo::DFloat.zeros(n_outputs * n_features)
        bounds = Numo::DFloat.zeros(n_outputs * n_features, 2)
        bounds.shape[0].times { |n| bounds[n, 1] = Float::INFINITY }

        res = Numo::Optimize.minimize(
          fnc: method(:nnls_fnc), jcb: true, x_init: w_init, args: [x, y, @params[:reg_param]], bounds: bounds,
          maxiter: @params[:max_iter], factr: @params[:tol] / Numo::Optimize::Lbfgsb::DBL_EPSILON,
          verbose: @params[:verbose] ? 1 : -1
        )

        @n_iter = res[:n_iter]
        w = single_target?(y) ? res[:x] : res[:x].reshape(n_outputs, n_features)
        @weight_vec, @bias_term = split_weight(w)

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

      def nnls_fnc(w, x, y, alpha)
        n_samples, n_features = x.shape
        w = w.reshape(y.shape[1], n_features) unless y.shape[1].nil?
        z = x.dot(w.transpose)
        d = z - y
        loss = (d**2).sum.fdiv(n_samples) + alpha * (w * w).sum
        gradient = 2.fdiv(n_samples) * d.transpose.dot(x) + 2.0 * alpha * w
        [loss, gradient.flatten.dup]
      end

      def single_target?(y)
        y.ndim == 1
      end
    end
  end
end
