# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module Manifold
    # LocalTangentSpaceAlignment is a class that implements Local Tangent Space Alignment.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/manifold/local_tangent_space_alignment'
    #
    #   lem = Rumale::Manifold::LocalTangentSpaceAlignment.new(n_components: 2, n_neighbors: 15)
    #   z = lem.fit_transform(x)
    #
    # *Reference*
    # - Zhang, A., and Zha, H., "Principal Manifolds and Nonlinear Diemnsion Reduction via Local Tangent Space Alignment," SIAM Journal on Scientific Computing, vol. 26, iss. 1, pp. 313-338, 2004.
    class LocalTangentSpaceAlignment < Rumale::Base::Estimator
      include Rumale::Base::Transformer

      # Return the data in representation space.
      # @return [Numo::DFloat] (shape: [n_samples, n_components])
      attr_reader :embedding

      # Create a new transformer with Local Tangent Space Alignment.
      #
      # @param n_components [Integer] The number of dimensions on representation space.
      # @param n_neighbors [Integer] The number of nearest neighbors for finding k-nearest neighbors
      # @param reg_param [Float] The reguralization parameter for local gram matrix in transform method.
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
      # @overload fit(x) -> LocalTangentSpaceAlignment
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [LocalTangentSpaceAlignment] The learned transformer itself.
      def fit(x, _y = nil)
        unless enable_linalg?(warning: false)
          raise 'LocalTangentSpaceAlignment#fit requires Numo::Linalg but that is not loaded'
        end

        x = Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        distance_mat = Rumale::PairwiseMetric.squared_error(x)
        neighbor_ids = neighbor_ids(distance_mat, @params[:n_neighbors], true)

        affinity_mat = Numo::DFloat.zeros(n_samples, n_samples)
        x_tangent = Numo::DFloat.zeros(@params[:n_neighbors], @params[:n_components] + 1)
        x_tangent[true, 0] = 1.fdiv(Math.sqrt(@params[:n_neighbors]))

        n_samples.times do |n|
          x_local = x[neighbor_ids[n, true], true]
          x_tangent[true, 1...] = right_singular_vectors(x_local, @params[:n_components])
          weight_mat = x_tangent.dot(x_tangent.transpose)
          neighbor_ids[n, true].each_with_index do |m, i|
            affinity_mat[m, neighbor_ids[n, true]] -= weight_mat[i, true]
            affinity_mat[m, m] += 1
          end
        end

        kernel_mat = 0.5 * (affinity_mat.transpose + affinity_mat)
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
          raise 'LocalTangentSpaceAlignment#fit_transform requires Numo::Linalg but that is not loaded'
        end

        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      # For out-of-sample data embedding, the same method as Locally Linear Embedding is used.
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

      def right_singular_vectors(x_local, n_singulars)
        n_samples = x_local.shape[0]
        x_local -= x_local.mean(0)
        gram_mat = x_local.dot(x_local.transpose)
        _, evecs = Numo::Linalg.eigh(gram_mat, vals_range: (n_samples - n_singulars)...n_samples)
        evecs.reverse(1).dup
      end
    end
  end
end
