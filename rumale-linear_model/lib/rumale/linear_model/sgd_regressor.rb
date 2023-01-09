# frozen_string_literal: true

require 'rumale/base/regressor'
require 'rumale/validation'

require_relative 'sgd_estimator'

module Rumale
  module LinearModel
    # SGDRegressor is a class that implements linear regressor with stochastic gradient descent optimization.
    #
    # @example
    #   require 'rumale/linear_model/sgd_regressor'
    #
    #   estimator =
    #     Rumale::LinearModel::SGDRegressor.new(loss: 'squared_error', reg_param: 1.0, max_iter: 1000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_target_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Shalev-Shwartz, S., and Singer, Y., "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Proc. ICML'07, pp. 807--814, 2007.
    # - Tsuruoka, Y., Tsujii, J., and Ananiadou, S., "Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty," Proc. ACL'09, pp. 477--485, 2009.
    # - Bottou, L., "Large-Scale Machine Learning with Stochastic Gradient Descent," Proc. COMPSTAT'10, pp. 177--186, 2010.
    class SGDRegressor < Rumale::LinearModel::SGDEstimator
      include Rumale::Base::Regressor

      # Return the random generator for performing random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new linear regressor with stochastic gradient descent optimization.
      #
      # @param loss [String] The loss function to be used ('squared_error' and 'epsilon_insensitive').
      # @param learning_rate [Float] The initial value of learning rate.
      #   The learning rate decreases as the iteration proceeds according to the equation: learning_rate / (1 + decay * t).
      # @param decay [Float] The smoothing parameter for decreasing learning rate as the iteration proceeds.
      #   If nil is given, the decay sets to 'reg_param * learning_rate'.
      # @param momentum [Float] The momentum factor.
      # @param penalty [String] The regularization type to be used ('l1', 'l2', and 'elasticnet').
      # @param l1_ratio [Float] The elastic-net type regularization mixing parameter.
      #   If penalty set to 'l2' or 'l1', this parameter is ignored.
      #   If l1_ratio = 1, the regularization is similar to Lasso.
      #   If l1_ratio = 0, the regularization is similar to Ridge.
      #   If 0 < l1_ratio < 1, the regularization is a combination of L1 and L2.
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param epsilon [Float] The margin of tolerance. If loss set to 'squared_error', this parameter is ignored.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param batch_size [Integer] The size of the mini batches.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      # @param n_jobs [Integer] The number of jobs for running the fit method in parallel.
      #   If nil is given, the method does not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(loss: 'squared_error', learning_rate: 0.01, decay: nil, momentum: 0.9,
                     penalty: 'l2', reg_param: 1.0, l1_ratio: 0.5,
                     fit_bias: true, bias_scale: 1.0,
                     epsilon: 0.1,
                     max_iter: 1000, batch_size: 50, tol: 1e-4,
                     n_jobs: nil, verbose: false, random_seed: nil)
        super()
        @params.merge!(
          loss: loss,
          learning_rate: learning_rate,
          decay: decay,
          momentum: momentum,
          penalty: penalty,
          reg_param: reg_param,
          l1_ratio: l1_ratio,
          fit_bias: fit_bias,
          bias_scale: bias_scale,
          epsilon: epsilon,
          max_iter: max_iter,
          batch_size: batch_size,
          tol: tol,
          n_jobs: n_jobs,
          verbose: verbose,
          random_seed: random_seed
        )
        @params[:decay] ||= @params[:reg_param] * @params[:learning_rate]
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @penalty_type = @params[:penalty]
        @loss_func = case @params[:loss]
                     when Rumale::LinearModel::Loss::MeanSquaredError::NAME
                       Rumale::LinearModel::Loss::MeanSquaredError.new
                     when Rumale::LinearModel::Loss::EpsilonInsensitive::NAME
                       Rumale::LinearModel::Loss::EpsilonInsensitive.new(epsilon: @params[:epsilon])
                     else
                       raise ArgumentError, "given loss '#{loss}' is not supported."
                     end
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @retu:rn [SGDRegressor] The learned regressor itself.
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
        x = Rumale::Validation.check_convert_sample_array(x)

        x.dot(@weight_vec.transpose) + @bias_term
      end
    end
  end
end
