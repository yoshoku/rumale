# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'

module Rumale
  module Clustering
    # DBSCAN is a class that implements DBSCAN cluster analysis.
    # The current implementation uses the Euclidean distance for analyzing the clusters.
    #
    # @example
    #   analyzer = Rumale::Clustering::DBSCAN.new(eps: 0.5, min_samples: 5)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - M. Ester, H-P. Kriegel, J. Sander, and X. Xu, "A density-based algorithm for discovering clusters in large spatial databases with noise," Proc. KDD' 96, pp. 266--231, 1996.
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
      def initialize(eps: 0.5, min_samples: 5)
        check_params_float(eps: eps)
        check_params_integer(min_samples: min_samples)
        @params = {}
        @params[:eps] = eps
        @params[:min_samples] = min_samples
        @core_sample_ids = nil
        @labels = nil
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> DBSCAN
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      # @return [DBSCAN] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        check_sample_array(x)
        partial_fit(x)
        self
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        check_sample_array(x)
        partial_fit(x)
        labels
      end

      # Dump marshal data.
      # @return [Hash] The marshal data.
      def marshal_dump
        { params: @params,
          core_sample_ids: @core_sample_ids,
          labels: @labels }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @core_sample_ids = obj[:core_sample_ids]
        @labels = obj[:labels]
        nil
      end

      private

      def partial_fit(x)
        cluster_id = 0
        n_samples  = x.shape[0]
        @core_sample_ids = []
        @labels = Numo::Int32.zeros(n_samples) - 2
        n_samples.times do |q|
          next if @labels[q] >= -1
          cluster_id += 1 if expand_cluster(x, q, cluster_id)
        end
        @core_sample_ids = Numo::Int32[*@core_sample_ids.flatten]
        nil
      end

      def expand_cluster(x, query_id, cluster_id)
        target_ids = region_query(x[query_id, true], x)
        if target_ids.size < @params[:min_samples]
          @labels[query_id] = -1
          false
        else
          @labels[target_ids] = cluster_id
          @core_sample_ids.push(target_ids.dup)
          target_ids.delete(query_id)
          while (m = target_ids.shift)
            neighbor_ids = region_query(x[m, true], x)
            next if neighbor_ids.size < @params[:min_samples]
            neighbor_ids.each do |n|
              target_ids.push(n) if @labels[n] < -1
              @labels[n] = cluster_id if @labels[n] <= -1
            end
          end
          true
        end
      end

      def region_query(query, targets)
        distance_arr = PairwiseMetric.euclidean_distance(query.expand_dims(0), targets)[0, true]
        distance_arr.lt(@params[:eps]).where.to_a
      end
    end
  end
end
