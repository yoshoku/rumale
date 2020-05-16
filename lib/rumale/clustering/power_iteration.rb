# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'

module Rumale
  module Clustering
    # PowerIteration is a class that implements power iteration clustering.
    #
    # @example
    #   analyzer = Rumale::Clustering::PowerIteration.new(n_clusters: 10, gamma: 8.0, max_iter: 1000)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Lin, F., and Cohen, W W., "Power Iteration Clustering," Proc. ICML'10, pp. 655--662, 2010.
    class PowerIteration
      include Base::BaseEstimator
      include Base::ClusterAnalyzer

      # Return the data in embedded space.
      # @return [Numo::DFloat] (shape: [n_samples])
      attr_reader :embedding

      # Return the cluster labels.
      # @return [Numo::Int32] (shape: [n_samples])
      attr_reader :labels

      # Return the number of iterations run for optimization
      # @return [Integer]
      attr_reader :n_iter

      # Create a new cluster analyzer with power iteration clustering.
      #
      # @param n_clusters [Integer] The number of clusters.
      # @param affinity [String] The representation of affinity matrix ('rbf' or 'precomputed').
      # @param gamma [Float] The parameter of rbf kernel, if nil it is 1 / n_features.
      #   If affinity = 'precomputed', this parameter is ignored.
      # @param init [String] The initialization method for centroids of K-Means clustering ('random' or 'k-means++').
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param eps [Float] A small value close to zero to avoid zero division error.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_clusters: 8, affinity: 'rbf', gamma: nil, init: 'k-means++', max_iter: 1000, tol: 1.0e-8, eps: 1.0e-5, random_seed: nil)
        check_params_numeric(n_clusters: n_clusters, max_iter: max_iter, tol: tol, eps: eps)
        check_params_numeric_or_nil(gamma: gamma, random_seed: random_seed)
        check_params_string(affinity: affinity, init: init)
        check_params_positive(n_clusters: n_clusters, max_iter: max_iter, tol: tol, eps: eps)
        @params = {}
        @params[:n_clusters] = n_clusters
        @params[:affinity] = affinity
        @params[:gamma] = gamma
        @params[:init] = init == 'random' ? 'random' : 'k-means++'
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:eps] = eps
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @embedding = nil
        @labels = nil
        @n_iter = nil
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> PowerIteration
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square affinity matrix (shape: [n_samples, n_samples]).
      # @return [PowerIteration] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        raise ArgumentError, 'Expect the input affinity matrix to be square.' if @params[:affinity] == 'precomputed' && x.shape[0] != x.shape[1]

        fit_predict(x)
        self
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square affinity matrix (shape: [n_samples, n_samples]).
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = check_convert_sample_array(x)
        raise ArgumentError, 'Expect the input affinity matrix to be square.' if @params[:affinity] == 'precomputed' && x.shape[0] != x.shape[1]

        affinity_mat = @params[:metric] == 'precomputed' ? x : Rumale::PairwiseMetric.rbf_kernel(x, nil, @params[:gamma])
        @embedding, @n_iter = embedded_space(affinity_mat, @params[:max_iter], @params[:tol].fdiv(affinity_mat.shape[0]))
        @labels = line_kmeans_clustering(@embedding)
      end

      private

      def embedded_space(affinity_mat, max_iter, tol)
        affinity_mat[affinity_mat.diag_indices] = 0.0

        degrees = affinity_mat.sum(axis: 1)
        normalized_affinity_mat = (1.0 / degrees).diag.dot(affinity_mat)

        iters = 0
        embedded_line = degrees / degrees.sum
        n_samples = embedded_line.shape[0]
        error = Numo::DFloat.ones(n_samples)
        max_iter.times do |t|
          iters = t + 1
          new_embedded_line = normalized_affinity_mat.dot(embedded_line)
          new_embedded_line /= new_embedded_line.abs.sum
          new_error = (new_embedded_line - embedded_line).abs
          break if (new_error - error).abs.max <= tol

          embedded_line = new_embedded_line
          error = new_error
        end

        [embedded_line, iters]
      end

      def line_kmeans_clustering(vec)
        Rumale::Clustering::KMeans.new(
          n_clusters: @params[:n_clusters], init: @params[:init],
          max_iter: @params[:max_iter], tol: @params[:tol], random_seed: @params[:random_seed]
        ).fit_predict(vec.expand_dims(1))
      end
    end
  end
end
