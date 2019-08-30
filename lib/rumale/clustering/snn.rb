# frozen_string_literal: true

require 'rumale/pairwise_metric'
require 'rumale/clustering/dbscan'

module Rumale
  module Clustering
    # SNN is a class that implements Shared Nearest Neighbor cluster analysis.
    # The SNN method is a variation of DBSCAN that uses similarity based on k-nearest neighbors as a metric.
    #
    # @example
    #   analyzer = Rumale::Clustering::SNN.new(n_neighbros: 10, eps: 5, min_samples: 5)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - L. Ertoz, M. Steinbach, and V. Kumar, "Finding Clusters of Different Sizes, Shapes, and Densities in Noisy, High Dimensional Data," Proc. SDM'03, pp. 47--58, 2003.
    # - M E. Houle, H-P. Kriegel, P. Kroger, E. Schubert, and A. Zimek, "Can Shared-Neighbor Distances Defeat the Curse of Dimensionality?," Proc. SSDBM'10, pp. 482--500, 2010.
    class SNN < DBSCAN
      # Create a new cluster analyzer with Shared Neareset Neighbor method.
      #
      # @param n_neighbors [Integer] The number of neighbors to be used for finding k-nearest neighbors.
      # @param eps [Integer] The threshold value for finding connected components based on similarity.
      # @param min_samples [Integer] The number of neighbor samples to be used for the criterion whether a point is a core point.
      # @param metric [String] The metric to calculate the distances.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance between points.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      def initialize(n_neighbors: 10, eps: 5, min_samples: 5, metric: 'euclidean')
        check_params_integer(n_neighbors: n_neighbors, min_samples: min_samples)
        check_params_string(metric: metric)
        @params = {}
        @params[:n_neighbors] = n_neighbors
        @params[:eps] = eps
        @params[:min_samples] = min_samples
        @params[:metric] = metric == 'precomputed' ? 'precomputed' : 'euclidean'
        @core_sample_ids = nil
        @labels = nil
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> SNN
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #     If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [SNN] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        super
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        super
      end

      private

      def calc_pairwise_metrics(x)
        distance_mat = @params[:metric] == 'precomputed' ? x : Rumale::PairwiseMetric.euclidean_distance(x)
        n_samples = distance_mat.shape[0]
        adjacency_mat = Numo::DFloat.zeros(n_samples, n_samples)
        n_samples.times do |n|
          neighbor_ids = distance_mat[n, true].sort_index[0...@params[:n_neighbors]]
          adjacency_mat[n, neighbor_ids] = 1
        end
        adjacency_mat.dot(adjacency_mat.transpose)
      end

      def region_query(similarity_arr)
        similarity_arr.gt(@params[:eps]).where.to_a
      end
    end
  end
end
