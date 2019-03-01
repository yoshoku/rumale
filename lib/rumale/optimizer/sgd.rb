# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/base_estimator'

module Rumale
  module Optimizer
    # SGD is a class that implements SGD optimizer.
    #
    # @example
    #   optimizer = Rumale::Optimizer::SGD.new(learning_rate: 0.01, momentum: 0.9, decay: 0.9)
    #   estimator = Rumale::LinearModel::LinearRegression.new(optimizer: optimizer, random_seed: 1)
    #   estimator.fit(samples, values)
    class SGD
      include Base::BaseEstimator
      include Validation

      # Create a new optimizer with SGD.
      #
      # @param learning_rate [Float] The initial value of learning rate.
      # @param momentum [Float] The initial value of momentum.
      # @param decay [Float] The smooting parameter.
      def initialize(learning_rate: 0.01, momentum: 0.0, decay: 0.0)
        check_params_float(learning_rate: learning_rate, momentum: momentum, decay: decay)
        check_params_positive(learning_rate: learning_rate, momentum: momentum, decay: decay)
        @params = {}
        @params[:learning_rate] = learning_rate
        @params[:momentum] = momentum
        @params[:decay] = decay
        @iter = 0
        @update = nil
      end

      # Calculate the updated weight with SGD.
      #
      # @param weight [Numo::DFloat] (shape: [n_features]) The weight to be updated.
      # @param gradient [Numo::DFloat] (shape: [n_features]) The gradient for updating the weight.
      # @return [Numo::DFloat] (shape: [n_feautres]) The updated weight.
      def call(weight, gradient)
        @update ||= Numo::DFloat.zeros(weight.shape[0])
        current_learning_rate = @params[:learning_rate] / (1.0 + @params[:decay] * @iter)
        @iter += 1
        @update = @params[:momentum] * @update - current_learning_rate * gradient
        weight + @update
      end

      # Dump marshal data.
      # @return [Hash] The marshal data.
      def marshal_dump
        { params: @params,
          iter: @iter,
          update: @update }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @iter = obj[:iter]
        @update = obj[:update]
        nil
      end
    end
  end
end
