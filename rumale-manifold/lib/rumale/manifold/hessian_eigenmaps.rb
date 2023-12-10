# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module Manifold
    # HessianEigenmaps is a class that implements Hessian Eigenmaps.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/manifold/hessian_eigenmaps'
    #
    #   hem = Rumale::Manifold::HessianEigenmaps.new(n_components: 2, n_neighbors: 15)
    #   z = hem.fit_transform(x)
    #
    # *Reference*
    # - Donoho, D. L., and Grimes, C., "Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data," Proc. Natl. Acad. Sci. USA, vol. 100, no. 10, pp. 5591--5596, 2003.
    class HessianEigenmaps < Rumale::Base::Estimator
      include Rumale::Base::Transformer

      # Return the data in representation space.
      # @return [Numo::DFloat] (shape: [n_samples, n_components])
      attr_reader :embedding

      # Create a new transformer with Hessian Eigenmaps.
      #
      # @param n_components [Integer] The number of dimensions on representation space.
      # @param n_neighbors [Integer] The number of nearest neighbors for k-nearest neighbor graph construction.
      # @param reg_param [Float] The reguralization parameter for local gram matrix in transform method.
      def initialize(n_neighbors: 5, n_components: 2, reg_param: 1e-6)
        super()
        @params = {
          n_neighbors: n_neighbors,
          n_components: n_components,
          reg_param: reg_param
        }
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> LocallyLinearEmbedding
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [LocallyLinearEmbedding] The learned transformer itself.
      def fit(x, _y = nil) # rubocop:disable Metrics/AbcSize
        raise 'HessianEigenmaps#fit requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        x = Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        distance_mat = Rumale::PairwiseMetric.squared_error(x)
        neighbor_ids = neighbor_ids(distance_mat, @params[:n_neighbors], true)

        tri_n_components = @params[:n_components] * (@params[:n_components] + 1) / 2
        hessian_mat = Numo::DFloat.zeros(n_samples * tri_n_components, n_samples)
        ones = Numo::DFloat.ones(@params[:n_neighbors], 1)
        n_samples.times do |i|
          tan_coords = tangent_coordinates(x[neighbor_ids[i, true], true])
          xi = Numo::DFloat.zeros(@params[:n_neighbors], tri_n_components)
          @params[:n_components].times do |m|
            offset = Array.new(m + 1) { |v| v }.sum
            (@params[:n_components] - m).times do |n|
              xi[true, m * @params[:n_components] - offset + n] = tan_coords[true, m] * tan_coords[true, m + n]
            end
          end

          xt, = Numo::Linalg.qr(Numo::DFloat.hstack([ones, tan_coords, xi]))
          pii = xt[true, (@params[:n_components] + 1)..-1]
          tri_n_components.times do |j|
            pj_sum = pii[true, j].sum
            normalizer = pj_sum <= 1e-8 ? 1 : 1.fdiv(pj_sum)
            hessian_mat[i * tri_n_components + j, neighbor_ids[i, true]] = pii[true, j] * normalizer
          end
        end

        kernel_mat = hessian_mat.transpose.dot(hessian_mat)
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
          raise 'HessianEigenmaps#fit_transform requires Numo::Linalg but that is not loaded'
        end

        fit(x)

        @embedding.dup
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

      def tangent_coordinates(x)
        m = x.mean(axis: 0)
        cx = x - m
        cov_mat = cx.transpose.dot(cx)
        n_features = x.shape[1]
        _, evecs = Numo::Linalg.eigh(cov_mat, vals_range: (n_features - @params[:n_components])...n_features)
        cx.dot(evecs.reverse(1))
      end
    end
  end
end
