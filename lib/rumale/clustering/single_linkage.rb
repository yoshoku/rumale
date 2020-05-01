# frozen_string_literal: true

require 'ostruct'
require 'rumale/base/base_estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'

module Rumale
  module Clustering
    # SingleLinkage is a class that implements hierarchical cluster analysis with single linakge method.
    # This class is used internally for HDBSCAN.
    #
    # @example
    #   analyzer = Rumale::Clustering::SingleLinkage.new(n_clusters: 2)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Mullner, D., "Modern hierarchical, agglomerative clustering algorithms," arXiv:1109.2378, 2011.
    class SingleLinkage
      include Base::BaseEstimator
      include Base::ClusterAnalyzer

      # Return the cluster labels.
      # @return [Numo::Int32] (shape: [n_samples])
      attr_reader :labels

      # Return the hierarchical structure.
      # @return [Array<OpenStruct>] (shape: [n_samples - 1])
      attr_reader :hierarchy

      # Create a new cluster analyzer with single linkage algorithm.
      #
      # @param n_clusters [Integer] The number of clusters.
      # @param metric [String] The metric to calculate the distances.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance between points.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      def initialize(n_clusters: 2, metric: 'euclidean')
        check_params_numeric(n_clusters: n_clusters)
        check_params_string(metric: metric)
        @params = {}
        @params[:n_clusters] = n_clusters
        @params[:metric] = metric == 'precomputed' ? 'precomputed' : 'euclidean'
        @labels = nil
        @hierarchy = nil
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> SingleLinkage
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [SingleLinkage] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        raise ArgumentError, 'Expect the input distance matrix to be square.' if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
        fit_predict(x)
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
        distance_mat = @params[:metric] == 'precomputed' ? x : Rumale::PairwiseMetric.euclidean_distance(x)
        @labels = partial_fit(distance_mat)
      end

      private

      # @!visibility private
      class UnionFind
        def initialize(n)
          @parent = Numo::Int32.zeros(2 * n - 1) - 1
          @size = Numo::Int32.hstack([Numo::Int32.ones(n), Numo::Int32.zeros(n - 1)])
          @next_label = n
        end

        # @!visibility private
        def union(x, y)
          size = @size[x] + @size[y]
          @parent[x] = @next_label
          @parent[y] = @next_label
          @size[@next_label] = size
          @next_label += 1
          size
        end

        # @!visibility private
        def find(x)
          p = x
          x = @parent[x] while @parent[x] != -1
          while @parent[p] != x
            p = @parent[p]
            @parent[p] = x
          end
          x
        end
      end

      private_constant :UnionFind

      def partial_fit(distance_mat)
        mst = minimum_spanning_tree(distance_mat)
        @hierarchy = single_linkage_hierarchy(mst)
        flatten(@hierarchy, @params[:n_clusters])
      end

      def minimum_spanning_tree(complete_graph)
        n_samples = complete_graph.shape[0]
        n_edges = n_samples - 1
        curr_weights = Numo::DFloat.zeros(n_samples) + Float::INFINITY
        curr_labels = Numo::Int32.new(n_samples).seq
        next_node = 0
        mst = Array.new(n_edges) do
          curr_node = next_node
          target = curr_labels.ne(curr_node)
          curr_labels = curr_labels[target]
          curr_weights = Numo::DFloat.minimum(curr_weights[target], complete_graph[curr_node, curr_labels])
          next_node = curr_labels[curr_weights.min_index]
          weight = curr_weights.min
          OpenStruct.new(x: curr_node, y: next_node, weight: weight)
        end
        mst.sort! { |a, b| a.weight <=> b.weight }
      end

      def single_linkage_hierarchy(mst)
        n_edges = mst.size
        n_nodes = n_edges + 1
        uf = UnionFind.new(n_nodes)
        Array.new(n_edges) do |n|
          x_root = uf.find(mst[n].x)
          y_root = uf.find(mst[n].y)
          x_root, y_root = [y_root, x_root] unless x_root < y_root
          weight = mst[n].weight
          n_samples = uf.union(x_root, y_root)
          OpenStruct.new(x: x_root, y: y_root, weight: weight, n_elements: n_samples)
        end
      end

      def descedent_ids(hierarchy_, start_node)
        n_samples = hierarchy_.size + 1
        return [start_node] if start_node < n_samples

        res = []
        indices = [start_node]
        n_indices = 1
        while n_indices.positive?
          idx = indices.pop
          if idx < n_samples
            res.push(idx)
            n_indices -= 1
          else
            indices.push(hierarchy_[idx - n_samples].x)
            indices.push(hierarchy_[idx - n_samples].y)
            n_indices += 1
          end
        end
        res
      end

      def flatten(hierarchy_, n_clusters)
        n_samples = hierarchy_.size + 1
        return Numo::Int32.zeros(n_samples) if n_clusters < 2

        nodes = [-([hierarchy_[-1].x, hierarchy_[-1].y].max + 1)]
        (n_clusters - 1).times do
          children = hierarchy_[-nodes[0] - n_samples]
          nodes.push(-children.x)
          nodes.push(-children.y)
          nodes.sort!.shift
        end
        res = Numo::Int32.zeros(n_samples)
        nodes.each_with_index { |sid, cluster_id| res[descedent_ids(hierarchy_, -sid)] = cluster_id }
        res
      end
    end
  end
end
