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
        y_not_given = y.nil?
        y = x if y_not_given
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y) unless y_not_given
        sum_x_vec = (x**2).sum(1).expand_dims(1)
        sum_y_vec = y_not_given ? sum_x_vec.transpose : (y**2).sum(1).expand_dims(1).transpose
        err_mat = -2 * x.dot(y.transpose)
        err_mat += sum_x_vec
        err_mat += sum_y_vec
        err_mat.class.maximum(err_mat, 0)
      end

      # Calculate the pairwise cosine simlarities between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def cosine_similarity(x, y = nil)
        y_not_given = y.nil?
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y) unless y_not_given
        x_norm = Numo::NMath.sqrt((x**2).sum(1))
        x_norm[x_norm.eq(0)] = 1
        x /= x_norm.expand_dims(1)
        if y_not_given
          x.dot(x.transpose)
        else
          y_norm = Numo::NMath.sqrt((y**2).sum(1))
          y_norm[y_norm.eq(0)] = 1
          y /= y_norm.expand_dims(1)
          x.dot(y.transpose)
        end
      end

      # Calculate the rbf kernel between x and y.
      #
      # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
      # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
      # @param gamma [Float] The parameter of rbf kernel, if nil it is 1 / n_features.
      # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def rbf_kernel(x, y = nil, gamma = nil)
        y_not_given = y.nil?
        y = x if y_not_given
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_sample_array(y) unless y_not_given
        gamma ||= 1.0 / x.shape[1]
        Rumale::Validation.check_params_numeric(gamma: gamma)
        Numo::NMath.exp(-gamma * squared_error(x, y))
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
