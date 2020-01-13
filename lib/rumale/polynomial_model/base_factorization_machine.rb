# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/optimizer/nadam'

module Rumale
  # This module consists of the classes that implement polynomial models.
  module PolynomialModel
    # BaseFactorizationMachine is an abstract class for implementation of Factorization Machine-based estimators.
    # This class is used internally.
    class BaseFactorizationMachine
      include Base::BaseEstimator

      # Initialize a Factorization Machine-based estimator.
      #
      # @param n_factors [Integer] The maximum number of iterations.
      # @param loss [String] The loss function ('hinge' or 'logistic' or nil).
      # @param reg_param_linear [Float] The regularization parameter for linear model.
      # @param reg_param_factor [Float] The regularization parameter for factor matrix.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param batch_size [Integer] The size of the mini batches.
      # @param optimizer [Optimizer] The optimizer to calculate adaptive learning rate.
      #   If nil is given, Nadam is used.
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2, loss: nil, reg_param_linear: 1.0, reg_param_factor: 1.0,
                     max_iter: 200, batch_size: 50, optimizer: nil, n_jobs: nil, random_seed: nil)
        @params = {}
        @params[:n_factors] = n_factors
        @params[:loss] = loss unless loss.nil?
        @params[:reg_param_linear] = reg_param_linear
        @params[:reg_param_factor] = reg_param_factor
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:optimizer] = optimizer
        @params[:optimizer] ||= Optimizer::Nadam.new
        @params[:n_jobs] = n_jobs
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @factor_mat = nil
        @weight_vec = nil
        @bias_term = nil
        @rng = Random.new(@params[:random_seed])
      end

      private

      def partial_fit(x, y)
        # Initialize some variables.
        n_samples, n_features = x.shape
        sub_rng = @rng.dup
        weight_vec = Numo::DFloat.zeros(n_features + 1)
        factor_mat = Numo::DFloat.zeros(@params[:n_factors], n_features)
        weight_optimizer = @params[:optimizer].dup
        factor_optimizers = Array.new(@params[:n_factors]) { @params[:optimizer].dup }
        # Start optimization.
        @params[:max_iter].times do |_t|
          sample_ids = [*0...n_samples]
          sample_ids.shuffle!(random: sub_rng)
          until (subset_ids = sample_ids.shift(@params[:batch_size])).empty?
            # Sampling.
            sub_x = x[subset_ids, true]
            sub_y = y[subset_ids]
            ex_sub_x = expand_feature(sub_x)
            # Calculate gradients for loss function.
            loss_grad = loss_gradient(sub_x, ex_sub_x, sub_y, factor_mat, weight_vec)
            next if loss_grad.ne(0.0).count.zero?
            # Update each parameter.
            weight_vec = weight_optimizer.call(weight_vec, weight_gradient(loss_grad, ex_sub_x, weight_vec))
            @params[:n_factors].times do |n|
              factor_mat[n, true] = factor_optimizers[n].call(factor_mat[n, true],
                                                              factor_gradient(loss_grad, sub_x, factor_mat[n, true]))
            end
          end
        end
        [factor_mat, *split_weight_vec_bias(weight_vec)]
      end

      def loss_gradient(_x, _expanded_x, _y, _factor, _weight)
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      def weight_gradient(loss_grad, data, weight)
        (loss_grad.expand_dims(1) * data).mean(0) + @params[:reg_param_linear] * weight
      end

      def factor_gradient(loss_grad, data, factor)
        (loss_grad.expand_dims(1) * (data * data.dot(factor).expand_dims(1) - factor * (data**2))).mean(0) +
          @params[:reg_param_factor] * factor
      end

      def expand_feature(x)
        Numo::NArray.hstack([x, Numo::DFloat.ones([x.shape[0], 1])])
      end

      def split_weight_vec_bias(weight_vec)
        weights = weight_vec[0...-1].dup
        bias = weight_vec[-1]
        [weights, bias]
      end
    end
  end
end
