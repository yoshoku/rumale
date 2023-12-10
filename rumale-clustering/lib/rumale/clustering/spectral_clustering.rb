# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'
require 'rumale/utils'
require 'rumale/validation'
require 'rumale/clustering/k_means'

module Rumale
  module Clustering
    # SpectralClustering is a class that implements the normalized spectral clustering.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/clustering/spectral_clustering'
    #
    #   analyzer = Rumale::Clustering::SpectralClustering.new(n_clusters: 10, gamma: 8.0)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Ng, A Y., Jordan, M I., and Weiss, Y., "On Spectral Clustering: Analyssi and an algorithm," Proc. NIPS'01, pp. 849--856, 2001.
    # - von Luxburg, U., "A tutorial on spectral clustering," Statistics and Computing, Vol. 17 (4), pp. 395--416, 2007.
    class SpectralClustering < ::Rumale::Base::Estimator
      include ::Rumale::Base::ClusterAnalyzer

      # Return the data in embedded space.
      # @return [Numo::DFloat] (shape: [n_samples, n_clusters])
      attr_reader :embedding

      # Return the cluster labels.
      # @return [Numo::Int32] (shape: [n_samples])
      attr_reader :labels

      # Create a new cluster analyzer with normalized spectral clustering.
      #
      # @param n_clusters [Integer] The number of clusters.
      # @param affinity [String] The representation of affinity matrix ('rbf' or 'precomputed').
      #   If affinity = 'rbf', the class performs the normalized spectral clustering with the fully connected graph weighted by rbf kernel.
      # @param gamma [Float] The parameter of rbf kernel, if nil it is 1 / n_features.
      #   If affinity = 'precomputed', this parameter is ignored.
      # @param init [String] The initialization method for centroids of K-Means clustering ('random' or 'k-means++').
      # @param max_iter [Integer] The maximum number of iterations for K-Means clustering.
      # @param tol [Float] The tolerance of termination criterion for K-Means clustering.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_clusters: 2, affinity: 'rbf', gamma: nil, init: 'k-means++', max_iter: 10, tol: 1.0e-8, random_seed: nil)
        super()
        @params = {
          n_clusters: n_clusters,
          affinity: affinity,
          gamma: gamma,
          init: (init == 'random' ? 'random' : 'k-means++'),
          max_iter: max_iter,
          tol: tol,
          random_seed: random_seed || srand
        }
      end

      # Analysis clusters with given training data.
      # To execute this method, Numo::Linalg must be loaded.
      #
      # @overload fit(x) -> SpectralClustering
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #     If the metric is 'precomputed', x must be a square affinity matrix (shape: [n_samples, n_samples]).
      #   @return [SpectralClustering] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        raise ArgumentError, 'the input affinity matrix should be square' if check_invalid_array_shape(x)

        raise 'SpectralClustering#fit requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        fit_predict(x)
        self
      end

      # Analysis clusters and assign samples to clusters.
      # To execute this method, Numo::Linalg must be loaded.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square affinity matrix (shape: [n_samples, n_samples]).
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        raise ArgumentError, 'the input affinity matrix should be square' if check_invalid_array_shape(x)

        unless enable_linalg?(warning: false)
          raise 'SpectralClustering#fit_predict requires Numo::Linalg but that is not loaded'
        end

        affinity_mat = @params[:metric] == 'precomputed' ? x : ::Rumale::PairwiseMetric.rbf_kernel(x, nil, @params[:gamma])
        @embedding = embedded_space(affinity_mat, @params[:n_clusters])
        normalized_embedding = ::Rumale::Utils.normalize(@embedding, 'l2')
        @labels = kmeans_clustering(normalized_embedding)
      end

      private

      def check_invalid_array_shape(x)
        @params[:affinity] == 'precomputed' && x.shape[0] != x.shape[1]
      end

      def embedded_space(affinity_mat, n_clusters)
        affinity_mat[affinity_mat.diag_indices] = 0.0
        degrees = 1.0 / Numo::NMath.sqrt(affinity_mat.sum(axis: 1))
        laplacian_mat = degrees.diag.dot(affinity_mat).dot(degrees.diag)

        n_samples = affinity_mat.shape[0]
        _, eig_vecs = Numo::Linalg.eigh(laplacian_mat, vals_range: (n_samples - n_clusters)...n_samples)
        eig_vecs.reverse(1).dup
      end

      def kmeans_clustering(x)
        ::Rumale::Clustering::KMeans.new(
          n_clusters: @params[:n_clusters], init: @params[:init],
          max_iter: @params[:max_iter], tol: @params[:tol], random_seed: @params[:random_seed]
        ).fit_predict(x)
      end
    end
  end
end
