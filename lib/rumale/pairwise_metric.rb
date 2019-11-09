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
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y)
        Numo::NMath.sqrt(squared_error(x, y).abs)
      end

      # Calculate the pairwise manhattan distances between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def manhattan_distance(x, y = nil)
        y = x if y.nil?
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y)
        n_samples_x = x.shape[0]
        n_samples_y = y.shape[0]
        distance_mat = Numo::DFloat.zeros(n_samples_x, n_samples_y)
        n_samples_x.times do |n|
          distance_mat[n, true] = (y - x[n, true]).abs.sum(axis: 1)
        end
        distance_mat
      end

      # Calculate the pairwise squared errors between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def squared_error(x, y = nil)
        y = x if y.nil?
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y)
        n_features = x.shape[1]
        one_vec = Numo::DFloat.ones(n_features).expand_dims(1)
        sum_x_vec = (x**2).dot(one_vec)
        sum_y_vec = (y**2).dot(one_vec).transpose
        dot_xy_mat = x.dot(y.transpose)
        dot_xy_mat * -2.0 + sum_x_vec + sum_y_vec
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
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y)
        Rumale::Validation.check_params_numeric(gamma: gamma)
        Numo::NMath.exp(-gamma * squared_error(x, y).abs)
      end

      # Calculate the linear kernel between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def linear_kernel(x, y = nil)
        y = x if y.nil?
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y)
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
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y)
        Rumale::Validation.check_params_numeric(gamma: gamma, degree: degree, coef: coef)
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
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y)
        Rumale::Validation.check_params_numeric(gamma: gamma, coef: coef)
        Numo::NMath.tanh(x.dot(y.transpose) * gamma + coef)
      end
    end
  end
end
