# frozen_string_literal: true

require 'numo/narray'

module Rumale
  # Module for calculating pairwise distances, similarities, and kernels.
  module PairwiseMetric
    module_function

    # Calculate the pairwise euclidean distances between x and y.
    #
    # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
    # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
    # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
    def euclidean_distance(x, y = nil)
      Numo::NMath.sqrt(squared_error(x, y).abs)
    end

    # Calculate the pairwise manhattan distances between x and y.
    #
    # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
    # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
    # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
    def manhattan_distance(x, y = nil)
      y = x if y.nil?
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
      sum_x_vec = (x**2).sum(axis: 1).expand_dims(1)
      sum_y_vec = y.nil? ? sum_x_vec.transpose : (y**2).sum(axis: 1).expand_dims(1).transpose
      y = x if y.nil?
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
      x_norm = Numo::NMath.sqrt((x**2).sum(axis: 1))
      x_norm[x_norm.eq(0)] = 1
      x /= x_norm.expand_dims(1)
      if y.nil?
        x.dot(x.transpose)
      else
        y_norm = Numo::NMath.sqrt((y**2).sum(axis: 1))
        y_norm[y_norm.eq(0)] = 1
        y /= y_norm.expand_dims(1)
        x.dot(y.transpose)
      end
    end

    # Calculate the pairwise cosine distances between x and y.
    #
    # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
    # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
    # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
    def cosine_distance(x, y = nil)
      dist_mat = 1 - cosine_similarity(x, y)
      dist_mat[dist_mat.diag_indices] = 0 if y.nil?
      dist_mat.clip(0, 2)
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
      Numo::NMath.exp(-gamma * squared_error(x, y))
    end

    # Calculate the linear kernel between x and y.
    #
    # @param x [Numo::DFloat] (shape: [n_samples_x, n_features])
    # @param y [Numo::DFloat] (shape: [n_samples_y, n_features])
    # @return [Numo::DFloat] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
    def linear_kernel(x, y = nil)
      y = x if y.nil?
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
    def polynomial_kernel(x, y = nil, degree = 3, gamma = nil, coef = 1) # rubocop:disable Metrics/ParameterLists
      y = x if y.nil?
      gamma ||= 1.0 / x.shape[1]
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
      Numo::NMath.tanh(x.dot(y.transpose) * gamma + coef)
    end
  end
end
