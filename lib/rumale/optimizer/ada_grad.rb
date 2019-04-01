# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/base_estimator'

module Rumale
  module Optimizer
    # AdaGrad is a class that implements AdaGrad optimizer.
    #
    # @example
    #   optimizer = Rumale::Optimizer::AdaGrad.new(learning_rate: 0.01, momentum: 0.9)
    #   estimator = Rumale::LinearModel::LinearRegression.new(optimizer: optimizer, random_seed: 1)
    #   estimator.fit(samples, values)
    #
    # *Reference*
    # - J. Duchi, E Hazan, and Y. Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization," J. Machine Learning Research, vol. 12, pp. 2121--2159, 2011.
    class AdaGrad
      include Base::BaseEstimator
      include Validation

      # Create a new optimizer with AdaGrad.
      #
      # @param learning_rate [Float] The initial value of learning rate.
      def initialize(learning_rate: 0.01)
        check_params_float(learning_rate: learning_rate)
        check_params_positive(learning_rate: learning_rate)
        @params = {}
        @params[:learning_rate] = learning_rate
        @moment = nil
      end

      # Calculate the updated weight with AdaGrad adaptive learning rate.
      #
      # @param weight [Numo::DFloat] (shape: [n_features]) The weight to be updated.
      # @param gradient [Numo::DFloat] (shape: [n_features]) The gradient for updating the weight.
      # @return [Numo::DFloat] (shape: [n_feautres]) The updated weight.
      def call(weight, gradient)
        @moment ||= Numo::DFloat.zeros(weight.shape[0])
        @moment += gradient**2
        weight - (@params[:learning_rate] / (@moment**0.5 + 1.0e-8)) * gradient
      end

      # Dump marshal data.
      # @return [Hash] The marshal data.
      def marshal_dump
        { params: @params,
          moment: @moment }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @moment = obj[:moment]
        nil
      end
    end
  end
end
