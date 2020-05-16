# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/base_estimator'

module Rumale
  # This module consists of the classes that implement optimizers adaptively tuning hyperparameters.
  #
  # @deprecated Optimizer module will be deleted in version 0.20.0.
  module Optimizer
    # Nadam is a class that implements Nadam optimizer.
    #
    # @deprecated Nadam will be deleted in version 0.20.0.
    #
    # *Reference*
    # - Dozat, T., "Incorporating Nesterov Momentum into Adam," Tech. Repo. Stanford University, 2015.
    class Nadam
      include Base::BaseEstimator
      include Validation

      # Create a new optimizer with Nadam
      #
      # @param learning_rate [Float] The initial value of learning rate.
      # @param decay1 [Float] The smoothing parameter for the first moment.
      # @param decay2 [Float] The smoothing parameter for the second moment.
      def initialize(learning_rate: 0.01, decay1: 0.9, decay2: 0.999)
        warn 'warning: Nadam is deprecated. This class will be deleted in version 0.20.0.'
        check_params_numeric(learning_rate: learning_rate, decay1: decay1, decay2: decay2)
        check_params_positive(learning_rate: learning_rate, decay1: decay1, decay2: decay2)
        @params = {}
        @params[:learning_rate] = learning_rate
        @params[:decay1] = decay1
        @params[:decay2] = decay2
        @fst_moment = nil
        @sec_moment = nil
        @decay1_prod = 1.0
        @iter = 0
      end

      # Calculate the updated weight with Nadam adaptive learning rate.
      #
      # @param weight [Numo::DFloat] (shape: [n_features]) The weight to be updated.
      # @param gradient [Numo::DFloat] (shape: [n_features]) The gradient for updating the weight.
      # @return [Numo::DFloat] (shape: [n_feautres]) The updated weight.
      def call(weight, gradient)
        @fst_moment ||= Numo::DFloat.zeros(weight.shape[0])
        @sec_moment ||= Numo::DFloat.zeros(weight.shape[0])

        @iter += 1

        decay1_curr = @params[:decay1] * (1.0 - 0.5 * 0.96**(@iter * 0.004))
        decay1_next = @params[:decay1] * (1.0 - 0.5 * 0.96**((@iter + 1) * 0.004))
        decay1_prod_curr = @decay1_prod * decay1_curr
        decay1_prod_next = @decay1_prod * decay1_curr * decay1_next
        @decay1_prod = decay1_prod_curr

        @fst_moment = @params[:decay1] * @fst_moment + (1.0 - @params[:decay1]) * gradient
        @sec_moment = @params[:decay2] * @sec_moment + (1.0 - @params[:decay2]) * gradient**2
        nm_gradient = gradient / (1.0 - decay1_prod_curr)
        nm_fst_moment = @fst_moment / (1.0 - decay1_prod_next)
        nm_sec_moment = @sec_moment / (1.0 - @params[:decay2]**@iter)

        weight - (@params[:learning_rate] / (nm_sec_moment**0.5 + 1e-8)) * ((1 - decay1_curr) * nm_gradient + decay1_next * nm_fst_moment)
      end
    end
  end
end
