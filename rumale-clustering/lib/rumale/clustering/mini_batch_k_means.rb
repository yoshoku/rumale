# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module Clustering
    # MniBatchKMeans is a class that implements K-Means cluster analysis
    # with mini-batch stochastic gradient descent (SGD).
    #
    # @example
    #   require 'rumale/clustering/mini_batch_k_means'
    #
    #   analyzer = Rumale::Clustering::MiniBatchKMeans.new(n_clusters: 10, max_iter: 50, batch_size: 50, random_seed: 1)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Sculley, D., "Web-scale k-means clustering," Proc. WWW'10, pp. 1177--1178, 2010.
    class MiniBatchKMeans < ::Rumale::Base::Estimator
      include ::Rumale::Base::ClusterAnalyzer

      # Return the centroids.
      # @return [Numo::DFloat] (shape: [n_clusters, n_features])
      attr_reader :cluster_centers

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new cluster analyzer with K-Means method with mini-batch SGD.
      #
      # @param n_clusters [Integer] The number of clusters.
      # @param init [String] The initialization method for centroids ('random' or 'k-means++').
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param tol [Float] The tolerance of termination criterion.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_clusters: 8, init: 'k-means++', max_iter: 100, batch_size: 100, tol: 1.0e-4, random_seed: nil)
        super()
        @params = {
          n_clusters: n_clusters,
          init: (init == 'random' ? 'random' : 'k-means++'),
          max_iter: max_iter,
          batch_size: batch_size,
          tol: tol,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> MiniBatchKMeans
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   @return [KMeans] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        # initialization.
        n_samples = x.shape[0]
        update_counter = Numo::Int32.zeros(@params[:n_clusters])
        sub_rng = @rng.dup
        init_cluster_centers(x, sub_rng)
        # optimization with mini-batch sgd.
        @params[:max_iter].times do |_t|
          sample_ids = Array(0...n_samples).shuffle(random: sub_rng)
          old_centers = @cluster_centers.dup
          until (subset_ids = sample_ids.shift(@params[:batch_size])).empty?
            # sub sampling
            sub_x = x[subset_ids, true]
            # assign nearest centroids
            cluster_labels = assign_cluster(sub_x)
            # update centroids
            @params[:n_clusters].times do |c|
              assigned_bits = cluster_labels.eq(c)
              next unless assigned_bits.count.positive?

              update_counter[c] += 1
              learning_rate = 1.fdiv(update_counter[c])
              update = sub_x[assigned_bits.where, true].mean(axis: 0)
              @cluster_centers[c, true] = (1 - learning_rate) * @cluster_centers[c, true] + learning_rate * update
            end
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
        x = ::Rumale::Validation.check_convert_sample_array(x)

        assign_cluster(x)
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit(x).predict(x)
      end

      private

      def assign_cluster(x)
        distance_matrix = ::Rumale::PairwiseMetric.euclidean_distance(x, @cluster_centers)
        distance_matrix.min_index(axis: 1) - Numo::Int32[*0.step(distance_matrix.size - 1, @cluster_centers.shape[0])]
      end

      def init_cluster_centers(x, sub_rng)
        # random initialize
        n_samples = x.shape[0]
        rand_id = Array(0...n_samples).sample(@params[:n_clusters], random: sub_rng)
        @cluster_centers = x[rand_id, true].dup
        return unless @params[:init] == 'k-means++'

        # k-means++ initialize
        (1...@params[:n_clusters]).each do |n|
          distance_matrix = ::Rumale::PairwiseMetric.euclidean_distance(x, @cluster_centers[0...n, true])
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
