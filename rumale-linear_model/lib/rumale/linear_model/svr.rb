# frozen_string_literal: true

require 'numo/optimize'

require 'rumale/base/regressor'
require 'rumale/validation'

require_relative 'base_estimator'

module Rumale
  module LinearModel
    # SVR is a class that implements Support Vector Regressor with the squared epsilon-insensitive loss.
    #
    # @note
    #   Rumale::SVM provides linear and kernel support vector regressor based on LIBLINEAR and LIBSVM.
    #   If you prefer execution speed, you should use Rumale::SVM::LinearSVR.
    #   https://github.com/yoshoku/rumale-svm
    #
    # @example
    #   require 'rumale/linear_model/svr'
    #
    #   estimator = Rumale::LinearModel::SVR.new(reg_param: 1.0, epsilon: 0.1)
    #   estimator.fit(training_samples, traininig_target_values)
    #   results = estimator.predict(testing_samples)
    class SVR < Rumale::LinearModel::BaseEstimator
      include Rumale::Base::Regressor

      # Create a new regressor with Support Vector Machine by the SGD optimization.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param epsilon [Float] The margin of tolerance.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      # @param n_jobs [Integer] The number of jobs for running the fit method in parallel.
      #   If nil is given, the method does not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      def initialize(reg_param: 1.0, fit_bias: true, bias_scale: 1.0, epsilon: 0.1, max_iter: 1000, tol: 1e-4,
                     n_jobs: nil, verbose: false)
        super()
        @params = {
          reg_param: reg_param,
          fit_bias: fit_bias,
          bias_scale: bias_scale,
          epsilon: epsilon,
          max_iter: max_iter,
          tol: tol,
          n_jobs: n_jobs,
          verbose: verbose
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [SVR] The learned regressor itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_target_value_array(y)
        Rumale::Validation.check_sample_size(x, y)

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

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        x.dot(@weight_vec.transpose) + @bias_term
      end

      private

      def partial_fit(base_x, single_y)
        fnc = proc do |w, x, y, eps, reg_param|
          n_samples = x.shape[0]
          z = x.dot(w)
          d = y - z
          loss = 0.5 * reg_param * w.dot(w) + (x.class.maximum(0, d.abs - eps)**2).sum.fdiv(n_samples)
          c = x.class.zeros(n_samples)
          indices = d.gt(eps)
          c[indices] = -d[indices] + eps if indices.count.positive?
          indices = d.lt(eps)
          c[indices] = -d[indices] - eps if indices.count.positive?
          grad = reg_param * w + 2.fdiv(n_samples) * x.transpose.dot(c)
          [loss, grad]
        end

        base_x = expand_feature(base_x) if fit_bias?

        n_features = base_x.shape[1]
        w_init = Numo::DFloat.zeros(n_features)

        res = Numo::Optimize.minimize(
          fnc: fnc, jcb: true, x_init: w_init, args: [base_x, single_y, @params[:epsilon], @params[:reg_param]],
          maxiter: @params[:max_iter], factr: @params[:tol] / Numo::Optimize::Lbfgsb::DBL_EPSILON,
          verbose: @params[:verbose] ? 1 : -1
        )

        split_weight(res[:x])
      end
    end
  end
end
