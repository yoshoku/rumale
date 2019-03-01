# frozen_string_literal: true

require 'rumale/validation'

module Rumale
  # Module for calculating pairwise distances, similarities, and kernels.
  module PairwiseMetric
    class << self
      # Calculate the pairwise euclidean distances between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def euclidean_distance(x, y = nil)
        y = x if y.nil?
        Rumale::Validation.check_sample_array(x)
        Rumale::Validation.check_sample_array(y)
        sum_x_vec = (x**2).sum(1)
        sum_y_vec = (y**2).sum(1)
        dot_xy_mat = x.dot(y.transpose)
        distance_matrix = dot_xy_mat * -2.0 +
                          sum_x_vec.tile(y.shape[0], 1).transpose +
                          sum_y_vec.tile(x.shape[0], 1)
        Numo::NMath.sqrt(distance_matrix.abs)
      end

      # Calculate the rbf kernel between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @param gamma [Float] The parameter of rbf kernel, if nil it is 1 / n_features.
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def rbf_kernel(x, y = nil, gamma = nil)
        y = x if y.nil?
        gamma ||= 1.0 / x.shape[1]
        Rumale::Validation.check_sample_array(x)
        Rumale::Validation.check_sample_array(y)
        Rumale::Validation.check_params_float(gamma: gamma)
        distance_matrix = euclidean_distance(x, y)
        Numo::NMath.exp((distance_matrix**2) * -gamma)
      end

      # Calculate the linear kernel between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def linear_kernel(x, y = nil)
        y = x if y.nil?
        Rumale::Validation.check_sample_array(x)
        Rumale::Validation.check_sample_array(y)
        x.dot(y.transpose)
      end

      # Calculate the polynomial kernel between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @param degree [Integer] The parameter of polynomial kernel.
      # @param gamma [Float] The parameter of polynomial kernel, if nil it is 1 / n_features.
      # @param coef [Integer] The parameter of polynomial kernel.
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def polynomial_kernel(x, y = nil, degree = 3, gamma = nil, coef = 1)
        y = x if y.nil?
        gamma ||= 1.0 / x.shape[1]
        Rumale::Validation.check_sample_array(x)
        Rumale::Validation.check_sample_array(y)
        Rumale::Validation.check_params_float(gamma: gamma)
        Rumale::Validation.check_params_integer(degree: degree, coef: coef)
        (x.dot(y.transpose) * gamma + coef)**degree
      end

      # Calculate the sigmoid kernel between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @param gamma [Float] The parameter of polynomial kernel, if nil it is 1 / n_features.
      # @param coef [Integer] The parameter of polynomial kernel.
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def sigmoid_kernel(x, y = nil, gamma = nil, coef = 1)
        y = x if y.nil?
        gamma ||= 1.0 / x.shape[1]
        Rumale::Validation.check_sample_array(x)
        Rumale::Validation.check_sample_array(y)
        Rumale::Validation.check_params_float(gamma: gamma)
        Rumale::Validation.check_params_integer(coef: coef)
        Numo::NMath.tanh(x.dot(y.transpose) * gamma + coef)
      end
    end
  end
end
