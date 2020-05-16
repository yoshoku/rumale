# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'

module Rumale
  # This module consists of classes that implement cluster analysis methods.
  module Clustering
    # KMeans is a class that implements K-Means cluster analysis.
    # The current implementation uses the Euclidean distance for analyzing the clusters.
    #
    # @example
    #   analyzer = Rumale::Clustering::KMeans.new(n_clusters: 10, max_iter: 50)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Arthur, D., and Vassilvitskii, S., "k-means++: the advantages of careful seeding," Proc. SODA'07, pp. 1027--1035, 2007.
    class KMeans
      include Base::BaseEstimator
      include Base::ClusterAnalyzer

      # Return the centroids.
      # @return [Numo::DFloat] (shape: [n_clusters, n_features])
      attr_reader :cluster_centers

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new cluster analyzer with K-Means method.
      #
      # @param n_clusters [Integer] The number of clusters.
      # @param init [String] The initialization method for centroids ('random' or 'k-means++').
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_clusters: 8, init: 'k-means++', max_iter: 50, tol: 1.0e-4, random_seed: nil)
        check_params_numeric(n_clusters: n_clusters, max_iter: max_iter, tol: tol)
        check_params_string(init: init)
        check_params_numeric_or_nil(random_seed: random_seed)
        check_params_positive(n_clusters: n_clusters, max_iter: max_iter)
        @params = {}
        @params[:n_clusters] = n_clusters
        @params[:init] = init == 'random' ? 'random' : 'k-means++'
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @cluster_centers = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> KMeans
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      # @return [KMeans] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        init_cluster_centers(x)
        @params[:max_iter].times do |_t|
          cluster_labels = assign_cluster(x)
          old_centers = @cluster_centers.dup
          @params[:n_clusters].times do |n|
            assigned_bits = cluster_labels.eq(n)
            @cluster_centers[n, true] = x[assigned_bits.where, true].mean(axis: 0) if assigned_bits.count.positive?
          end
          error = Numo::NMath.sqrt(((old_centers - @cluster_centers)**2).sum(axis: 1)).mean
          break if error <= @params[:tol]
        end
        self
      end

      # Predict cluster labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the cluster label.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        assign_cluster(x)
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = check_convert_sample_array(x)
        fit(x)
        predict(x)
      end

      private

      def assign_cluster(x)
        distance_matrix = PairwiseMetric.euclidean_distance(x, @cluster_centers)
        distance_matrix.min_index(axis: 1) - Numo::Int32[*0.step(distance_matrix.size - 1, @cluster_centers.shape[0])]
      end

      def init_cluster_centers(x)
        # random initialize
        n_samples = x.shape[0]
        sub_rng = @rng.dup
        rand_id = [*0...n_samples].sample(@params[:n_clusters], random: sub_rng)
        @cluster_centers = x[rand_id, true].dup
        return unless @params[:init] == 'k-means++'

        # k-means++ initialize
        (1...@params[:n_clusters]).each do |n|
          distance_matrix = PairwiseMetric.euclidean_distance(x, @cluster_centers[0...n, true])
          min_distances = distance_matrix.flatten[distance_matrix.min_index(axis: 1)]
          probs = min_distances**2 / (min_distances**2).sum
          cum_probs = probs.cumsum
          selected_id = cum_probs.gt(sub_rng.rand).where.to_a.first
          @cluster_centers[n, true] = x[selected_id, true].dup
        end
      end
    end
  end
end
