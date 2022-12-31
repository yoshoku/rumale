# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/validation'

module Rumale
  module KernelMachine
    # KernelRidge is a class that implements kernel ridge regression.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/pairwise_metric'
    #   require 'rumale/kernel_machine/kernel_ridge'
    #
    #   kernel_mat_train = Rumale::PairwiseMetric::rbf_kernel(training_samples)
    #   kridge = Rumale::KernelMachine::KernelRidge.new(reg_param: 1.0)
    #   kridge.fit(kernel_mat_train, traininig_values)
    #
    #   kernel_mat_test = Rumale::PairwiseMetric::rbf_kernel(test_samples, training_samples)
    #   results = kridge.predict(kernel_mat_test)
    class KernelRidge < ::Rumale::Base::Estimator
      include ::Rumale::Base::Regressor

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_training_sample, n_outputs])
      attr_reader :weight_vec

      # Create a new regressor with kernel ridge regression.
      #
      # @param reg_param [Float/Numo::DFloat] The regularization parameter.
      def initialize(reg_param: 1.0)
        super()
        @params = {
          reg_param: reg_param
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_training_samples, n_training_samples])
      #   The kernel matrix of the training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The taget values to be used for fitting the model.
      # @return [KernelRidge] The learned regressor itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        raise ArgumentError, 'Expect the kernel matrix of training data to be square.' unless x.shape[0] == x.shape[1]
        raise 'KernelRidge#fit requires Numo::Linalg but that is not loaded.' unless enable_linalg?(warning: false)

        n_samples = x.shape[0]

        if @params[:reg_param].is_a?(Float)
          reg_kernel_mat = x + Numo::DFloat.eye(n_samples) * @params[:reg_param]
          @weight_vec = Numo::Linalg.solve(reg_kernel_mat, y, driver: 'sym')
        else
          n_outputs = y.shape[1]
          @weight_vec = Numo::DFloat.zeros(n_samples, n_outputs)
          n_outputs.times do |n|
            reg_kernel_mat = x + Numo::DFloat.eye(n_samples) * @params[:reg_param][n]
            @weight_vec[true, n] = Numo::Linalg.solve(reg_kernel_mat, y[true, n], driver: 'sym')
          end
        end

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to predict values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        x.dot(@weight_vec)
      end
    end
  end
end
