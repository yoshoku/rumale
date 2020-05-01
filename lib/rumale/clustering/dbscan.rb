# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'

module Rumale
  module Clustering
    # DBSCAN is a class that implements DBSCAN cluster analysis.
    #
    # @example
    #   analyzer = Rumale::Clustering::DBSCAN.new(eps: 0.5, min_samples: 5)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Ester, M., Kriegel, H-P., Sander, J., and Xu, X., "A density-based algorithm for discovering clusters in large spatial databases with noise," Proc. KDD' 96, pp. 266--231, 1996.
    class DBSCAN
      include Base::BaseEstimator
      include Base::ClusterAnalyzer

      # Return the core sample indices.
      # @return [Numo::Int32] (shape: [n_core_samples])
      attr_reader :core_sample_ids

      # Return the cluster labels. The negative cluster label indicates that the point is noise.
      # @return [Numo::Int32] (shape: [n_samples])
      attr_reader :labels

      # Create a new cluster analyzer with DBSCAN method.
      #
      # @param eps [Float] The radius of neighborhood.
      # @param min_samples [Integer] The number of neighbor samples to be used for the criterion whether a point is a core point.
      # @param metric [String] The metric to calculate the distances.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance between points.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      def initialize(eps: 0.5, min_samples: 5, metric: 'euclidean')
        check_params_numeric(eps: eps, min_samples: min_samples)
        check_params_string(metric: metric)
        @params = {}
        @params[:eps] = eps
        @params[:min_samples] = min_samples
        @params[:metric] = metric == 'precomputed' ? 'precomputed' : 'euclidean'
        @core_sample_ids = nil
        @labels = nil
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> DBSCAN
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [DBSCAN] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        raise ArgumentError, 'Expect the input distance matrix to be square.' if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
        partial_fit(x)
        self
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = check_convert_sample_array(x)
        raise ArgumentError, 'Expect the input distance matrix to be square.' if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
        partial_fit(x)
        labels
      end

      private

      def partial_fit(x)
        cluster_id = 0
        metric_mat = calc_pairwise_metrics(x)
        n_samples = metric_mat.shape[0]
        @core_sample_ids = []
        @labels = Numo::Int32.zeros(n_samples) - 2
        n_samples.times do |query_id|
          next if @labels[query_id] >= -1
          cluster_id += 1 if expand_cluster(metric_mat, query_id, cluster_id)
        end
        @core_sample_ids = Numo::Int32[*@core_sample_ids.flatten]
        nil
      end

      def calc_pairwise_metrics(x)
        @params[:metric] == 'precomputed' ? x : Rumale::PairwiseMetric.euclidean_distance(x)
      end

      def expand_cluster(metric_mat, query_id, cluster_id)
        target_ids = region_query(metric_mat[query_id, true])
        if target_ids.size < @params[:min_samples]
          @labels[query_id] = -1
          false
        else
          @labels[target_ids] = cluster_id
          @core_sample_ids.push(target_ids.dup)
          target_ids.delete(query_id)
          while (m = target_ids.shift)
            neighbor_ids = region_query(metric_mat[m, true])
            next if neighbor_ids.size < @params[:min_samples]
            neighbor_ids.each do |n|
              target_ids.push(n) if @labels[n] < -1
              @labels[n] = cluster_id if @labels[n] <= -1
            end
          end
          true
        end
      end

      def region_query(metric_arr)
        metric_arr.lt(@params[:eps]).where.to_a
      end
    end
  end
end
