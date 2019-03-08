# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/base_estimator'

module Rumale
  module Optimizer
    # Adam is a class that implements Adam optimizer.
    #
    # @example
    #   optimizer = Rumale::Optimizer::Adam.new(learning_rate: 0.01, momentum: 0.9, decay1: 0.9, decay2: 0.999)
    #   estimator = Rumale::LinearModel::LinearRegression.new(optimizer: optimizer, random_seed: 1)
    #   estimator.fit(samples, values)
    #
    # *Reference*
    # - D P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," Proc. ICLR'15, 2015.
    class Adam
      include Base::BaseEstimator
      include Validation

      # Create a new optimizer with Adam
      #
      # @param learning_rate [Float] The initial value of learning rate.
      # @param decay1 [Float] The smoothing parameter for the first moment.
      # @param decay2 [Float] The smoothing parameter for the second moment.
      def initialize(learning_rate: 0.001, decay1: 0.9, decay2: 0.999)
        check_params_float(learning_rate: learning_rate, decay1: decay1, decay2: decay2)
        check_params_positive(learning_rate: learning_rate, decay1: decay1, decay2: decay2)
        @params = {}
        @params[:learning_rate] = learning_rate
        @params[:decay1] = decay1
        @params[:decay2] = decay2
        @fst_moment = nil
        @sec_moment = nil
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

        @fst_moment = @params[:decay1] * @fst_moment + (1.0 - @params[:decay1]) * gradient
        @sec_moment = @params[:decay2] * @sec_moment + (1.0 - @params[:decay2]) * gradient**2
        nm_fst_moment = @fst_moment / (1.0 - @params[:decay1]**@iter)
        nm_sec_moment = @sec_moment / (1.0 - @params[:decay2]**@iter)

        weight - @params[:learning_rate] * nm_fst_moment / (nm_sec_moment**0.5 + 1e-8)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data.
      def marshal_dump
        { params: @params,
          fst_moment: @fst_moment,
          sec_moment: @sec_moment,
          iter: @iter }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @fst_moment = obj[:fst_moment]
        @sec_moment = obj[:sec_moment]
        @iter = obj[:iter]
        nil
      end
    end
  end
end
