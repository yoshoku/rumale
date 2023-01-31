# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module Manifold
    # LaplacianEigenmaps is a class that implements Laplacian Eigenmaps.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/manifold/laplacian_eigenmaps'
    #
    #   lem = Rumale::Manifold::LaplacianEigenmaps.new(n_components: 2, n_neighbors: 15)
    #   z = lem.fit_transform(x)
    #
    # *Reference*
    # - Belkin, M., and Niyogi, P., "Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering," Proc. NIPS'01, pp. 585--591, 2001.
    class LaplacianEigenmaps < Rumale::Base::Estimator
      include Rumale::Base::Transformer

      # Return the data in representation space.
      # @return [Numo::DFloat] (shape: [n_samples, n_components])
      attr_reader :embedding

      # Create a new transformer with Laplacian Eigenmaps.
      #
      # @param n_components [Integer] The number of dimensions on representation space.
      # @param gamma [Nil/Float] The parameter of RBF kernel. If nil is given, the weight of affinity matrix sets to 1.
      # @param n_neighbors [Integer] The number of nearest neighbors for k-nearest neighbor graph construction.
      def initialize(n_components: 2, gamma: nil, n_neighbors: 10)
        super()
        @params = {
          n_components: n_components,
          gamma: gamma,
          n_neighbors: [1, n_neighbors].max
        }
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> LaplacianEigenmaps
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [LaplacianEigenmaps] The learned transformer itself.
      def fit(x, _y = nil)
        raise 'LaplacianEigenmaps#fit requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        x = Rumale::Validation.check_convert_sample_array(x)

        distance_mat = Rumale::PairwiseMetric.squared_error(x)
        neighbor_graph = k_neighbor_graph(distance_mat, @params[:n_neighbors], true)
        affinity_mat = if @params[:gamma].nil?
                         neighbor_graph
                       else
                         neighbor_graph * Numo::NMath.exp(-@params[:gamma] * distance_mat)
                       end
        degree_mat = affinity_mat.sum(axis: 1).diag
        laplacian_mat = degree_mat - affinity_mat

        _, eig_vecs = Numo::Linalg.eigh(laplacian_mat, degree_mat, vals_range: 1...(1 + @params[:n_components]))

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
          raise 'LaplacianEigenmaps#fit_transform requires Numo::Linalg but that is not loaded'
        end

        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = Rumale::Validation.check_convert_sample_array(x)

        distance_mat = Rumale::PairwiseMetric.squared_error(x, @x_train)
        neighbor_graph = k_neighbor_graph(distance_mat, @params[:n_neighbors], false)
        affinity_mat = if @params[:gamma].nil?
                         neighbor_graph
                       else
                         neighbor_graph * Numo::NMath.exp(-@params[:gamma] * distance_mat)
                       end
        normalizer = Numo::NMath.sqrt(affinity_mat.mean * affinity_mat.mean(axis: 1))
        n_train_samples = @x_train.shape[0]
        weight_mat = 1.fdiv(n_train_samples) * (affinity_mat.transpose / normalizer).transpose
        weight_mat.dot(@embedding)
      end

      private

      def k_neighbor_graph(distance_mat, n_neighbors, contain_self)
        n_samples = distance_mat.shape[0]
        if contain_self
          neighbor_graph = Numo::DFloat.zeros(n_samples, n_samples)
          n_samples.times do |n|
            neighbor_ids = (distance_mat[n, true].sort_index.to_a - [n])[0...n_neighbors]
            neighbor_graph[n, neighbor_ids] = 1
          end
          Numo::DFloat.maximum(neighbor_graph, neighbor_graph.transpose)
        else
          neighbor_graph = Numo::DFloat.zeros(distance_mat.shape)
          n_samples.times do |n|
            neighbor_ids = distance_mat[n, true].sort_index.to_a[0...n_neighbors]
            neighbor_graph[n, neighbor_ids] = 1
          end
          neighbor_graph
        end
      end
    end
  end
end
