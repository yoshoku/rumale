# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/base_estimator'

module Rumale
  module Optimizer
    # YellowFin is a class that implements YellowFin optimizer.
    #
    # *Reference*
    # - Zhang, J., and Mitliagkas, I., "YellowFin and the Art of Momentum Tuning," CoRR abs/1706.03471, 2017.
    class YellowFin
      include Base::BaseEstimator
      include Validation

      # Create a new optimizer with YellowFin.
      #
      # @param learning_rate [Float] The initial value of learning rate.
      # @param momentum [Float] The initial value of momentum.
      # @param decay [Float] The smooting parameter.
      # @param window_width [Integer] The sliding window width for searching curvature range.
      def initialize(learning_rate: 0.01, momentum: 0.9, decay: 0.999, window_width: 20)
        check_params_numeric(learning_rate: learning_rate, momentum: momentum, decay: decay, window_width: window_width)
        check_params_positive(learning_rate: learning_rate, momentum: momentum, decay: decay, window_width: window_width)
        @params = {}
        @params[:learning_rate] = learning_rate
        @params[:momentum] = momentum
        @params[:decay] = decay
        @params[:window_width] = window_width
        @smth_learning_rate = learning_rate
        @smth_momentum = momentum
        @grad_norms = nil
        @grad_norm_min = 0.0
        @grad_norm_max = 0.0
        @grad_mean_sqr = 0.0
        @grad_mean = 0.0
        @grad_var = 0.0
        @grad_norm_mean = 0.0
        @curve_mean = 0.0
        @distance_mean = 0.0
        @update = nil
      end

      # Calculate the updated weight with adaptive momentum coefficient and learning rate.
      #
      # @param weight [Numo::DFloat] (shape: [n_features]) The weight to be updated.
      # @param gradient [Numo::DFloat] (shape: [n_features]) The gradient for updating the weight.
      # @return [Numo::DFloat] (shape: [n_feautres]) The updated weight.
      def call(weight, gradient)
        @update ||= Numo::DFloat.zeros(weight.shape[0])
        curvature_range(gradient)
        gradient_variance(gradient)
        distance_to_optimum(gradient)
        @smth_momentum = @params[:decay] * @smth_momentum + (1 - @params[:decay]) * current_momentum
        @smth_learning_rate = @params[:decay] * @smth_learning_rate + (1 - @params[:decay]) * current_learning_rate
        @update = @smth_momentum * @update - @smth_learning_rate * gradient
        weight + @update
      end

      private

      def current_momentum
        dr = Math.sqrt(@grad_norm_max / @grad_norm_min + 1.0e-8)
        [cubic_root**2, ((dr - 1) / (dr + 1))**2].max
      end

      def current_learning_rate
        (1.0 - Math.sqrt(@params[:momentum]))**2 / (@grad_norm_min + 1.0e-8)
      end

      def cubic_root
        p = (@distance_mean**2 * @grad_norm_min**2) / (2 * @grad_var + 1.0e-8)
        w3 = (-Math.sqrt(p**2 + 4.fdiv(27) * p**3) - p).fdiv(2)
        w = (w3 >= 0.0 ? 1 : -1) * w3.abs**1.fdiv(3)
        y = w - p / (3 * w + 1.0e-8)
        y + 1
      end

      def curvature_range(gradient)
        @grad_norms ||= []
        @grad_norms.push((gradient**2).sum)
        @grad_norms.shift(@grad_norms.size - @params[:window_width]) if @grad_norms.size > @params[:window_width]
        @grad_norm_min = @params[:decay] * @grad_norm_min + (1 - @params[:decay]) * @grad_norms.min
        @grad_norm_max = @params[:decay] * @grad_norm_max + (1 - @params[:decay]) * @grad_norms.max
      end

      def gradient_variance(gradient)
        @grad_mean_sqr = @params[:decay] * @grad_mean_sqr + (1 - @params[:decay]) * gradient**2
        @grad_mean = @params[:decay] * @grad_mean + (1 - @params[:decay]) * gradient
        @grad_var = (@grad_mean_sqr - @grad_mean**2).sum
      end

      def distance_to_optimum(gradient)
        grad_sqr = (gradient**2).sum
        @grad_norm_mean = @params[:decay] * @grad_norm_mean + (1 - @params[:decay]) * Math.sqrt(grad_sqr + 1.0e-8)
        @curve_mean = @params[:decay] * @curve_mean + (1 - @params[:decay]) * grad_sqr
        @distance_mean = @params[:decay] * @distance_mean + (1 - @params[:decay]) * (@grad_norm_mean / @curve_mean)
      end
    end
  end
end
