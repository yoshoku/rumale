# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module Manifold
    # LocallyLinearEmbedding is a class that implements Loccaly Linear Embedding.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/manifold/locally_linear_embedding'
    #
    #   lem = Rumale::Manifold::LocallyLinearEmbedding.new(n_components: 2, n_neighbors: 15)
    #   z = lem.fit_transform(x)
    #
    # *Reference*
    # - Roweis, S., and Saul, L., "Nonlinear Dimensionality Reduction by Locally Linear Embedding," J. of Science, vol. 290, pp. 2323-2326, 2000.
    class LocallyLinearEmbedding < Rumale::Base::Estimator
      include Rumale::Base::Transformer

      # Return the data in representation space.
      # @return [Numo::DFloat] (shape: [n_samples, n_components])
      attr_reader :embedding

      # Create a new transformer with Locally Linear Embedding.
      #
      # @param n_components [Integer] The number of dimensions on representation space.
      # @param n_neighbors [Integer] The number of nearest neighbors for k-nearest neighbor graph construction.
      # @param reg_param [Float] The reguralization parameter for local gram matrix.
      def initialize(n_components: 2, n_neighbors: 10, reg_param: 1e-3)
        super()
        @params = {
          n_components: n_components,
          n_neighbors: [1, n_neighbors].max,
          reg_param: reg_param
        }
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> LocallyLinearEmbedding
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [LocallyLinearEmbedding] The learned transformer itself.
      def fit(x, _y = nil)
        raise 'LocallyLinearEmbedding#fit requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        x = Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        tol = @params[:reg_param].fdiv(@params[:n_neighbors])
        distance_mat = Rumale::PairwiseMetric.squared_error(x)
        neighbor_ids = neighbor_ids(distance_mat, @params[:n_neighbors], true)

        affinity_mat = Numo::DFloat.eye(n_samples)
        n_samples.times do |n|
          x_local = x[neighbor_ids[n, true], true] - x[n, true]
          gram_mat = x_local.dot(x_local.transpose)
          gram_mat += tol * gram_mat.trace * Numo::DFloat.eye(@params[:n_neighbors])
          weights = Numo::Linalg.solve(gram_mat, Numo::DFloat.ones(@params[:n_neighbors]))
          weights /= weights.sum + 1e-8
          affinity_mat[n, neighbor_ids[n, true]] -= weights
        end

        kernel_mat = affinity_mat.transpose.dot(affinity_mat)
        _, eig_vecs = Numo::Linalg.eigh(kernel_mat, vals_range: 1...(1 + @params[:n_components]))

        @embedding = @params[:n_components] == 1 ? eig_vecs[true, 0].dup : eig_vecs.dup
        @x_train = x.dup

        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _y = nil)
        unless enable_linalg?(warning: false)
          raise 'LocallyLinearEmbedding#fit_transform requires Numo::Linalg but that is not loaded'
        end

        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        tol = @params[:reg_param].fdiv(@params[:n_neighbors])
        distance_mat = Rumale::PairwiseMetric.squared_error(x, @x_train)
        neighbor_ids = neighbor_ids(distance_mat, @params[:n_neighbors], false)
        weight_mat = Numo::DFloat.zeros(n_samples, @x_train.shape[0])

        n_samples.times do |n|
          x_local = @x_train[neighbor_ids[n, true], true] - x[n, true]
          gram_mat = x_local.dot(x_local.transpose)
          gram_mat += tol * weight_mat.trace * Numo::DFloat.eye(@params[:n_neighbors])
          weights = Numo::Linalg.solve(gram_mat, Numo::DFloat.ones(@params[:n_neighbors]))
          weights /= weights.sum + 1e-8
          weight_mat[n, neighbor_ids[n, true]] = weights
        end

        weight_mat.dot(@embedding)
      end

      private

      def neighbor_ids(distance_mat, n_neighbors, contain_self)
        n_samples = distance_mat.shape[0]
        neighbor_ids = Numo::Int32.zeros(n_samples, n_neighbors)
        if contain_self
          n_samples.times { |n| neighbor_ids[n, true] = (distance_mat[n, true].sort_index.to_a - [n])[0...n_neighbors] }
        else
          n_samples.times { |n| neighbor_ids[n, true] = distance_mat[n, true].sort_index.to_a[0...n_neighbors] }
        end
        neighbor_ids
      end
    end
  end
end
