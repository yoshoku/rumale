# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/utils'

module Rumale
  module NeuralNetwork
    # BaseRVFL is an abstract class for implementation of random vector functional link (RVFL) network.
    # This class is used internally.
    #
    # *Reference*
    # - Malik, A. K., Gao, R., Ganaie, M. A., Tanveer, M., and Suganthan, P. N., "Random vector functional link network: recent developments, applications, and future directions," Applied Soft Computing, vol. 143, 2023.
    # - Zhang, L., and Suganthan, P. N., "A comprehensive evaluation of random vector functional link networks," Information Sciences, vol. 367--368, pp. 1094--1105, 2016.
    class BaseRVFL < ::Rumale::Base::Estimator
      # Create a random vector functional link network estimator.
      #
      # @param hidden_units [Array] The number of units in the hidden layer.
      # @param reg_param [Float] The regularization parameter.
      # @param scale [Float] The scale parameter for random weight and bias.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(hidden_units: 128, reg_param: 100.0, scale: 1.0, random_seed: nil)
        super()
        @params = {
          hidden_units: hidden_units,
          reg_param: reg_param,
          scale: scale,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
      end

      private

      def partial_fit(x, y)
        h = hidden_output(x)

        n_samples = h.shape[0]
        n_features = h.shape[1]
        reg_term = 1.fdiv(@params[:reg_param])

        @weight_vec = if n_features <= n_samples
                        Numo::Linalg.inv(h.transpose.dot(h) + reg_term * Numo::DFloat.eye(n_features)).dot(h.transpose).dot(y)
                      else
                        h.transpose.dot(Numo::Linalg.inv(h.dot(h.transpose) + reg_term * Numo::DFloat.eye(n_samples))).dot(y)
                      end
      end

      def hidden_output(x)
        sub_rng = @rng.dup
        n_features = x.shape[1]
        @random_weight_vec = (2.0 * Rumale::Utils.rand_uniform([n_features, @params[:hidden_units]], sub_rng) - 1.0) * @params[:scale] # rubocop:disbale Layout/LineLength
        @random_bias = Rumale::Utils.rand_uniform(@params[:hidden_units], sub_rng) * @params[:scale]
        h = 0.5 * (Numo::NMath.tanh(0.5 * (x.dot(@random_weight_vec) + @random_bias)) + 1.0)
        Numo::DFloat.hstack([x, h])
      end
    end
  end
end
