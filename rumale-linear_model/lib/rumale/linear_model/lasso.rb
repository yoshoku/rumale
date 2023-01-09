# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/validation'

require_relative 'base_estimator'

module Rumale
  module LinearModel
    # Lasso is a class that implements Lasso Regression with coordinate descent optimization.
    #
    # @example
    #   require 'rumale/linear_model/lasso'
    #
    #   estimator = Rumale::LinearModel::Lasso.new(reg_param: 0.1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Friedman, J., Hastie, T., and Tibshirani, R., "Regularization Paths for Generalized Linear Models via Coordinate Descent," Journal of Statistical Software, 33 (1), pp. 1--22, 2010.
    # - Simon, N., Friedman, J., and Hastie, T., "A Blockwise Descent Algorithm for Group-penalized Multiresponse and Multinomial Regression," arXiv preprint arXiv:1311.6529, 2013.
    class Lasso < Rumale::LinearModel::BaseEstimator
      include Rumale::Base::Regressor

      # Return the number of iterations performed in coordinate descent optimization.
      # @return [Integer]
      attr_reader :n_iter

      # Create a new Lasso regressor.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      def initialize(reg_param: 1.0, fit_bias: true, bias_scale: 1.0, max_iter: 1000, tol: 1e-4)
        super()
        @params = {
          reg_param: reg_param,
          fit_bias: fit_bias,
          bias_scale: bias_scale,
          max_iter: max_iter,
          tol: tol
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [Lasso] The learned regressor itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_target_value_array(y)
        Rumale::Validation.check_sample_size(x, y)

        @n_iter = 0
        x = expand_feature(x) if fit_bias?

        @weight_vec, @bias_term = if single_target?(y)
                                    partial_fit(x, y)
                                  else
                                    partial_fit_multi(x, y)
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

      def partial_fit(x, y)
        n_features = x.shape[1]
        w = Numo::DFloat.zeros(n_features)
        x_norms = (x**2).sum(axis: 0)
        residual = y - x.dot(w)

        @params[:max_iter].times do |iter|
          w_err = 0.0
          n_features.times do |j|
            next if x_norms[j].zero?

            w_prev = w[j]

            residual += w[j] * x[true, j]
            z = x[true, j].dot(residual)
            w[j] = soft_threshold(z, @params[:reg_param]).fdiv(x_norms[j])
            residual -= w[j] * x[true, j]

            w_err = [w_err, (w[j] - w_prev).abs].max
          end

          @n_iter = iter + 1

          break if w_err <= @params[:tol]
        end

        split_weight(w)
      end

      def partial_fit_multi(x, y)
        n_features = x.shape[1]
        n_outputs = y.shape[1]
        w = Numo::DFloat.zeros(n_outputs, n_features)
        x_norms = (x**2).sum(axis: 0)
        residual = y - x.dot(w.transpose)

        @params[:max_iter].times do |iter|
          w_err = 0.0
          n_features.times do |j|
            next if x_norms[j].zero?

            w_prev = w[true, j]

            residual += x[true, j].expand_dims(1) * w[true, j]
            z = x[true, j].dot(residual)
            w[true, j] = [1.0 - @params[:reg_param].fdiv(Math.sqrt((z**2).sum)), 0.0].max.fdiv(x_norms[j]) * z
            residual -= x[true, j].expand_dims(1) * w[true, j]

            w_err = [w_err, (w[true, j] - w_prev).abs.max].max
          end

          @n_iter = iter + 1

          break if w_err <= @params[:tol]
        end

        split_weight(w)
      end

      def soft_threshold(z, threshold)
        sign(z) * [z.abs - threshold, 0].max
      end

      def sign(z)
        return 0.0 if z.zero?

        z.positive? ? 1.0 : -1.0
      end

      def single_target?(y)
        y.ndim == 1
      end
    end
  end
end
