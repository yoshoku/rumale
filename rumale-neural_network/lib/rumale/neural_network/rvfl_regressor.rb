# frozen_string_literal: true

require 'rumale/base/regressor'
require 'rumale/neural_network/base_rvfl'
require 'rumale/validation'

module Rumale
  module NeuralNetwork
    # RVFLRegressor is a class that implements regressor based on random vector functional link (RVFL) network.
    # The current implementation uses sigmoid function as activation function.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #   Numo::Linalg = Numo::TinyLinalg
    #
    #   require 'rumale/neural_network/rvfl_regressor'
    #
    #   estimator = Rumale::NeuralNetwork::RVFLRegressor.new(hidden_units: 128, reg_param: 100.0)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Malik, A. K., Gao, R., Ganaie, M. A., Tanveer, M., and Suganthan, P. N., "Random vector functional link network: recent developments, applications, and future directions," Applied Soft Computing, vol. 143, 2023.
    # - Zhang, L., and Suganthan, P. N., "A comprehensive evaluation of random vector functional link networks," Information Sciences, vol. 367--368, pp. 1094--1105, 2016.
    class RVFLRegressor < BaseRVFL
      include ::Rumale::Base::Regressor

      # Return the weight vector in the hidden layer of RVFL network.
      # @return [Numo::DFloat] (shape: [n_hidden_units, n_features])
      attr_reader :random_weight_vec

      # Return the bias vector in the hidden layer of RVFL network.
      # @return [Numo::DFloat] (shape: [n_hidden_units])
      attr_reader :random_bias

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_features + n_hidden_units, n_outputs])
      attr_reader :weight_vec

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with RVFL network.
      #
      # @param hidden_units [Array] The number of units in the hidden layer.
      # @param reg_param [Float] The regularization parameter.
      # @param scale [Float] The scale parameter for random weight and bias.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(hidden_units: 128, reg_param: 100.0, scale: 1.0, random_seed: nil)
        super
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The taget values to be used for fitting the model.
      # @return [MLPRegressor] The learned regressor itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        raise 'RBFRegressor#fit requires Numo::Linalg but that is not loaded.' unless enable_linalg?(warning: false)

        y = y.expand_dims(1) if y.ndim == 1

        partial_fit(x, y)

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) The predicted values per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        h = hidden_output(x)
        out = h.dot(@weight_vec)
        out = out[true, 0] if out.shape[1] == 1
        out
      end
    end
  end
end
