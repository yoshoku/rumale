# frozen_string_literal: true

require 'rumale/base/regressor'
require 'rumale/neural_network/base_mlp'
require 'rumale/validation'

module Rumale
  module NeuralNetwork
    # MLPRegressor is a class that implements regressor based on multi-layer perceptron.
    # MLPRegressor use ReLu as the activation function and Adam as the optimization method
    # and mean squared error as the loss function.
    #
    # @example
    #   require 'rumale/neural_network/mlp_regressor'
    #
    #   estimator = Rumale::NeuralNetwork::MLPRegressor.new(hidden_units: [100, 100], dropout_rate: 0.3)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    class MLPRegressor < BaseMLP
      include ::Rumale::Base::Regressor

      # Return the network.
      # @return [Rumale::NeuralNetwork::Model::Sequential]
      attr_reader :network

      # Return the number of iterations run for optimization
      # @return [Integer]
      attr_reader :n_iter

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with multi-layer perceptron.
      #
      # @param hidden_units [Array] The number of units in the i-th hidden layer.
      # @param dropout_rate [Float] The rate of the units to drop.
      # @param learning_rate [Float] The initial value of learning rate in Adam optimizer.
      # @param decay1 [Float] The smoothing parameter for the first moment in Adam optimizer.
      # @param decay2 [Float] The smoothing parameter for the second moment in Adam optimizer.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param batch_size [Intger] The size of the mini batches.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(hidden_units: [128, 128], dropout_rate: 0.4, learning_rate: 0.001, decay1: 0.9, decay2: 0.999,
                     max_iter: 200, batch_size: 50, tol: 1e-4, verbose: false, random_seed: nil)
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

        y = y.expand_dims(1) if y.ndim == 1
        n_targets = y.shape[1]
        n_features = x.shape[1]
        sub_rng = @rng.dup

        loss = ::Rumale::NeuralNetwork::Loss::MeanSquaredError.new
        @network = buld_network(n_features, n_targets, sub_rng)
        @network = train(x, y, @network, loss, sub_rng)
        @network.delete_dropout

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        out, = @network.forward(x)
        out = out[true, 0] if out.shape[1] == 1
        out
      end
    end
  end
end
