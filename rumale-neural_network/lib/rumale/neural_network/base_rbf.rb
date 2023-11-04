# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/pairwise_metric'

module Rumale
  module NeuralNetwork
    # BaseRBF is an abstract class for implementation of radial basis function (RBF) network estimator.
    # This class is used internally.
    #
    # *Reference*
    # - Bugmann, G., "Normalized Gaussian Radial Basis Function networks," Neural Computation, vol. 20, pp. 97--110, 1998.
    # - Que, Q., and Belkin, M., "Back to the Future: Radial Basis Function Networks Revisited," Proc. of AISTATS'16, pp. 1375--1383, 2016.
    class BaseRBF < ::Rumale::Base::Estimator
      # Create a radial basis function network estimator.
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
        super()
        @params = {
          hidden_units: hidden_units,
          gamma: gamma,
          reg_param: reg_param,
          normalize: normalize,
          max_iter: max_iter,
          tol: tol,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
      end

      private

      def partial_fit(x, y)
        find_centers(x)

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
        h = ::Rumale::PairwiseMetric.rbf_kernel(x, @centers, @params[:gamma])
        h /= h.sum(axis: 1).expand_dims(1) if @params[:normalize]
        h
      end

      def find_centers(x)
        # initialize centers randomly
        n_samples = x.shape[0]
        sub_rng = @rng.dup
        rand_id = Array(0...n_samples).sample(n_centers, random: sub_rng)
        @centers = x[rand_id, true].dup

        # find centers
        @params[:max_iter].times do |_t|
          center_ids = assign_centers(x)
          old_centers = @centers.dup
          n_centers.times do |n|
            assigned_bits = center_ids.eq(n)
            @centers[n, true] = x[assigned_bits.where, true].mean(axis: 0) if assigned_bits.count.positive?
          end
          error = Numo::NMath.sqrt(((old_centers - @centers)**2).sum(axis: 1)).mean
          break if error <= @params[:tol]
        end
      end

      def assign_centers(x)
        distance_matrix = ::Rumale::PairwiseMetric.euclidean_distance(x, @centers)
        distance_matrix.min_index(axis: 1) - Numo::Int32[*0.step(distance_matrix.size - 1, @centers.shape[0])]
      end

      def n_centers
        @params[:hidden_units]
      end
    end
  end
end
