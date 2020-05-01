# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'

module Rumale
  module Clustering
    # KMedoids is a class that implements K-Medoids cluster analysis.
    #
    # @example
    #   analyzer = Rumale::Clustering::KMedoids.new(n_clusters: 10, max_iter: 50)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Arthur, D., and Vassilvitskii, S., "k-means++: the advantages of careful seeding," Proc. SODA'07, pp. 1027--1035, 2007.
    class KMedoids
      include Base::BaseEstimator
      include Base::ClusterAnalyzer

      # Return the indices of medoids.
      # @return [Numo::Int32] (shape: [n_clusters])
      attr_reader :medoid_ids

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new cluster analyzer with K-Medoids method.
      #
      # @param n_clusters [Integer] The number of clusters.
      # @param metric [String] The metric to calculate the distances.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance between points.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      # @param init [String] The initialization method for centroids ('random' or 'k-means++').
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_clusters: 8, metric: 'euclidean', init: 'k-means++', max_iter: 50, tol: 1.0e-4, random_seed: nil)
        check_params_numeric(n_clusters: n_clusters, max_iter: max_iter, tol: tol)
        check_params_string(metric: metric, init: init)
        check_params_numeric_or_nil(random_seed: random_seed)
        check_params_positive(n_clusters: n_clusters, max_iter: max_iter)
        @params = {}
        @params[:n_clusters] = n_clusters
        @params[:metric] = metric == 'precomputed' ? 'precomputed' : 'euclidean'
        @params[:init] = init == 'random' ? 'random' : 'k-means++'
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @medoid_ids = nil
        @cluster_centers = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> KMedoids
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [KMedoids] The learned cluster analyzer itself.
      def fit(x, _not_used = nil)
        x = check_convert_sample_array(x)
        raise ArgumentError, 'Expect the input distance matrix to be square.' if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
        # initialize some varibales.
        distance_mat = @params[:metric] == 'precomputed' ? x : Rumale::PairwiseMetric.euclidean_distance(x)
        init_cluster_centers(distance_mat)
        error = distance_mat[true, @medoid_ids].mean
        @params[:max_iter].times do |_t|
          cluster_labels = assign_cluster(distance_mat[true, @medoid_ids])
          @params[:n_clusters].times do |n|
            assigned_ids = cluster_labels.eq(n).where
            @medoid_ids[n] = assigned_ids[distance_mat[assigned_ids, assigned_ids].sum(axis: 1).min_index]
          end
          new_error = distance_mat[true, @medoid_ids].mean
          break if (error - new_error).abs <= @params[:tol]
          error = new_error
        end
        @cluster_centers = x[@medoid_ids, true].dup if @params[:metric] == 'euclidean'
        self
      end

      # Predict cluster labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the cluster label.
      #   If the metric is 'precomputed', x must be distances between samples and medoids (shape: [n_samples, n_clusters]).
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        distance_mat = @params[:metric] == 'precomputed' ? x : Rumale::PairwiseMetric.euclidean_distance(x, @cluster_centers)
        if @params[:metric] == 'precomputed' && distance_mat.shape[1] != @medoid_ids.size
          raise ArgumentError, 'Expect the size input matrix to be n_samples-by-n_clusters.'
        end
        assign_cluster(distance_mat)
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = check_convert_sample_array(x)
        fit(x)
        if @params[:metric] == 'precomputed'
          predict(x[true, @medoid_ids])
        else
          predict(x)
        end
      end

      private

      def assign_cluster(distances_to_medoids)
        distances_to_medoids.min_index(axis: 1) - Numo::Int32[*0.step(distances_to_medoids.size - 1, @params[:n_clusters])]
      end

      def init_cluster_centers(distance_mat)
        # random initialize
        n_samples = distance_mat.shape[0]
        sub_rng = @rng.dup
        @medoid_ids = Numo::Int32.asarray([*0...n_samples].sample(@params[:n_clusters], random: sub_rng))
        return unless @params[:init] == 'k-means++'
        # k-means++ initialize
        (1...@params[:n_clusters]).each do |n|
          distances = distance_mat[true, @medoid_ids[0...n]]
          min_distances = distances.flatten[distances.min_index(axis: 1)]
          probs = min_distances**2 / (min_distances**2).sum
          cum_probs = probs.cumsum
          @medoid_ids[n] = cum_probs.gt(sub_rng.rand).where.to_a.first
        end
      end
    end
  end
end
