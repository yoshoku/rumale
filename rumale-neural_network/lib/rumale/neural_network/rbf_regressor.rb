# frozen_string_literal: true

require 'rumale/base/regressor'
require 'rumale/neural_network/base_rbf'
require 'rumale/validation'

module Rumale
  module NeuralNetwork
    # RBFRegressor is a class that implements regressor based on (k-means) radial basis function (RBF) networks.
    #
    # @example
    #   require 'rumale/neural_network/rbf_regressor'
    #
    #   estimator = Rumale::NeuralNetwork::RBFRegressor.new(hidden_units: 128, reg_param: 100.0)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Bugmann, G., "Normalized Gaussian Radial Basis Function networks," Neural Computation, vol. 20, pp. 97--110, 1998.
    # - Que, Q., and Belkin, M., "Back to the Future: Radial Basis Function Networks Revisited," Proc. of AISTATS'16, pp. 1375--1383, 2016.
    class RBFRegressor < BaseRBF
      include ::Rumale::Base::Regressor

      # Return the centers in the hidden layer of RBF network.
      # @return [Numo::DFloat] (shape: [n_centers, n_features])
      attr_reader :centers

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_centers, n_outputs])
      attr_reader :weight_vec

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with (k-means) RBF networks.
      #
      # @param hidden_units [Array] The number of units in the hidden layer.
      # @param gamma [Float] The parameter for the radial basis function, if nil it is 1 / n_features.
      # @param reg_param [Float] The regularization parameter.
      # @param normalize [Boolean] The flag indicating whether to normalize the hidden layer output or not.
      # @param max_iter [Integer] The maximum number of iterations for finding centers.
      # @param tol [Float] The tolerance of termination criterion for finding centers.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(hidden_units: 128, gamma: nil, reg_param: 100.0, normalize: false,
                     max_iter: 50, tol: 1e-4, random_seed: nil)
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
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
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
