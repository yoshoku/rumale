# frozen_string_literal: true

require_relative 'base_estimator'

module Rumale
  module LinearModel
    # @!visibility private
    # This module consists of the classes that implement penalty (regularization) term.
    module Penalty
      # @!visibility private
      # L2Penalty is a class that applies L2 penalty to weight vector of linear model.
      # This class is used internally.
      class L2Penalty
        # @!visibility private
        def initialize(reg_param:)
          @reg_param = reg_param
        end

        # @!visibility private
        def call(weight, lr)
          weight - @reg_param * lr * weight
        end
      end

      # @!visibility private
      # L1Penalty is a class that applies L1 penalty to weight vector of linear model.
      # This class is used internally.
      class L1Penalty
        # @!visibility private
        def initialize(reg_param:)
          @u = 0.0
          @reg_param = reg_param
        end

        # @!visibility private
        def call(weight, lr)
          @q_vec ||= Numo::DFloat.zeros(weight.shape[0])
          @u += @reg_param * lr
          z = weight.dup
          gt = weight.gt(0)
          lt = weight.lt(0)
          weight[gt] = Numo::DFloat.maximum(0.0, weight[gt] - (@u + @q_vec[gt])) if gt.count.positive?
          weight[lt] = Numo::DFloat.minimum(0.0, weight[lt] + (@u - @q_vec[lt])) if lt.count.positive?
          @q_vec += weight - z
          weight
        end
      end
    end

    # @!visibility private
    # This module consists of the class that implements stochastic gradient descent (SGD) optimizer.
    module Optimizer
      # @!visibility private
      # SGD is a class that implements SGD optimizer.
      # This class is used internally.
      class SGD
        # @!visibility private
        # Create a new SGD optimizer.
        # @param learning_rate [Float] The initial value of learning rate.
        # @param momentum [Float] The initial value of momentum.
        # @param decay [Float] The smooting parameter.
        def initialize(learning_rate: 0.01, momentum: 0.0, decay: 0.0)
          @learning_rate = learning_rate
          @momentum = momentum
          @decay = decay
          @iter = 0
        end

        # @!visibility private
        def current_learning_rate
          @learning_rate / (1.0 + @decay * @iter)
        end

        # @!visibility private
        def call(weight, gradient)
          @update ||= Numo::DFloat.zeros(weight.shape[0])
          @update = @momentum * @update - current_learning_rate * gradient
          @iter += 1
          weight + @update
        end
      end
    end

    # @!visibility private
    # This module consists of the classes that implement loss function for linear model.
    module Loss
      # @!visibility private
      # MeanSquaredError is a class that calculates mean squared error for linear regression model.
      class MeanSquaredError
        # @!visibility private
        def loss(out, y)
          ((out - y)**2).sum.fdiv(y.shape[0])
        end

        # @!visibility private
        def dloss(out, y)
          2.fdiv(y.shape[0]) * (out - y)
        end

        # @!visibility private
        def name
          NAME
        end

        # @!visibility private
        NAME = 'squared_error'
      end

      # @!visibility private
      # LogLoss is a class that calculates logistic loss for logistic regression.
      class LogLoss
        # @!visibility private
        def loss(out, y)
          Numo::NMath.log(1 + Numo::NMath.exp(-y * out)).sum.fdiv(y.shape[0])
        end

        # @!visibility private
        def dloss(out, y)
          y / (1 + Numo::NMath.exp(-y * out)) - y
        end

        # @!visibility private
        def name
          NAME
        end

        # @!visibility private
        NAME = 'log_loss'
      end

      # @!visibility private
      # HingeLoss is a class that calculates hinge loss for support vector classifier.
      class HingeLoss
        # @!visibility private
        def loss(out, y)
          out.class.maximum(0.0, 1 - y * out).sum.fdiv(y.shape[0])
        end

        # @!visibility private
        def dloss(out, y)
          tids = (y * out).lt(1)
          d = Numo::DFloat.zeros(y.shape[0])
          d[tids] = -y[tids] if tids.count.positive?
          d
        end

        # @!visibility private
        def name
          NAME
        end

        # @!visibility private
        NAME = 'hinge'
      end

      # @!visibility private
      # EpsilonInsensitive is a class that calculates epsilon insensitive for support vector regressor.
      class EpsilonInsensitive
        # @!visibility private
        def initialize(epsilon: 0.1)
          @epsilon = epsilon
        end

        # @!visibility private
        def loss(out, y)
          out.class.maximum(0.0, (y - out).abs - @epsilon).sum.fdiv(y.shape[0])
        end

        # @!visibility private
        def dloss(out, y)
          d = Numo::DFloat.zeros(y.shape[0])
          tids = (out - y).gt(@epsilon)
          d[tids] = 1 if tids.count.positive?
          tids = (y - out).gt(@epsilon)
          d[tids] = -1 if tids.count.positive?
          d
        end

        # @!visibility private
        def name
          NAME
        end

        # @!visibility private
        NAME = 'epsilon_insensitive'
      end
    end

    # SGDEstimator is an abstract class for implementation of linear model with mini-batch stochastic gradient descent (SGD) optimization.
    # This class is used internally.
    class SGDEstimator < Rumale::LinearModel::BaseEstimator
      # Create an initial linear model with SGD.
      def initialize
        super
        @params = {
          learning_rate: 0.01,
          decay: nil,
          momentum: 0.0,
          bias_scale: 1.0,
          fit_bias: true,
          reg_param: 0.0,
          l1_ratio: 0.0,
          max_iter: 1000,
          batch_size: 50,
          tol: 0.0001,
          verbose: false
        }
      end

      private

      L2_PENALTY = 'l2'
      L1_PENALTY = 'l1'
      ELASTICNET_PENALTY = 'elasticnet'

      private_constant :L2_PENALTY, :L1_PENALTY, :ELASTICNET_PENALTY

      def init_vars(n_features)
        @sub_rng = @rng.dup
        @weight = Numo::DFloat.zeros(n_features)
        @optimizer = ::Rumale::LinearModel::Optimizer::SGD.new(
          learning_rate: @params[:learning_rate], momentum: @params[:momentum], decay: @params[:decay]
        )
        @l2_penalty = ::Rumale::LinearModel::Penalty::L2Penalty.new(reg_param: l2_reg_param)
        @l1_penalty = ::Rumale::LinearModel::Penalty::L1Penalty.new(reg_param: l1_reg_param)
      end

      def partial_fit_(x, y, max_iter: @params[:max_iter], init: true)
        class_name = self.class.to_s.split('::').last if @params[:verbose]
        narr = x.class
        # Expand feature vectors for bias term.
        x = expand_feature(x) if fit_bias?
        # Initialize some variables.
        n_samples, n_features = x.shape
        init_vars(n_features) if init
        # Optimization.
        max_iter.times do |t|
          sample_ids = Array(0...n_samples)
          sample_ids.shuffle!(random: @sub_rng)
          until (subset_ids = sample_ids.shift(@params[:batch_size])).empty?
            # sampling
            sub_x = x[subset_ids, true]
            sub_y = y[subset_ids]
            # calculate gradient
            dloss = @loss_func.dloss(sub_x.dot(@weight), sub_y)
            dloss = narr.minimum(1e12, narr.maximum(-1e12, dloss))
            gradient = dloss.dot(sub_x)
            # update weight
            lr = @optimizer.current_learning_rate
            @weight = @optimizer.call(@weight, gradient)
            # l2 regularization
            @weight = @l2_penalty.call(@weight, lr) if apply_l2_penalty?
            # l1 regularization
            @weight = @l1_penalty.call(@weight, lr) if apply_l1_penalty?
          end
          loss = @loss_func.loss(x.dot(@weight), y)
          puts "[#{class_name}] Loss after #{t + 1} epochs: #{loss}" if @params[:verbose]
          break if loss < @params[:tol]
        end
        split_weight(@weight)
      end

      def apply_l2_penalty?
        @penalty_type == L2_PENALTY || @penalty_type == ELASTICNET_PENALTY
      end

      def apply_l1_penalty?
        @penalty_type == L1_PENALTY || @penalty_type == ELASTICNET_PENALTY
      end

      def l2_reg_param
        case @penalty_type
        when ELASTICNET_PENALTY
          @params[:reg_param] * (1.0 - @params[:l1_ratio])
        when L2_PENALTY
          @params[:reg_param]
        else
          0.0
        end
      end

      def l1_reg_param
        case @penalty_type
        when ELASTICNET_PENALTY
          @params[:reg_param] * @params[:l1_ratio]
        when L1_PENALTY
          @params[:reg_param]
        else
          0.0
        end
      end
    end
  end
end
