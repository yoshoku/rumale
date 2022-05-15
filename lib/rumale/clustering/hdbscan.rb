# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'
require 'rumale/clustering/single_linkage'

module Rumale
  module Clustering
    # HDBSCAN is a class that implements HDBSCAN cluster analysis.
    #
    # @example
    #   analyzer = Rumale::Clustering::HDBSCAN.new(min_samples: 5)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Campello, R J. G. B., Moulavi, D., Zimek, A., and Sander, J., "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection," TKDD, Vol. 10 (1), pp. 5:1--5:51, 2015.
    # - Campello, R J. G. B., Moulavi, D., and Sander, J., "Density-Based Clustering Based on Hierarchical Density Estimates," Proc. PAKDD'13, pp. 160--172, 2013.
    # - Lelis, L., and Sander, J., "Semi-Supervised Density-Based Clustering," Proc. ICDM'09, pp. 842--847, 2009.
    class HDBSCAN
      include Base::BaseEstimator
      include Base::ClusterAnalyzer

      # Return the cluster labels. The negative cluster label indicates that the point is noise.
      # @return [Numo::Int32] (shape: [n_samples])
      attr_reader :labels

      # Create a new cluster analyzer with HDBSCAN algorithm.
      #
      # @param min_samples [Integer] The number of neighbor samples to be used for the criterion whether a point is a core point.
      # @param min_cluster_size [Integer/Nil] The minimum size of cluster. If nil is given, it is set equal to min_samples.
      # @param metric [String] The metric to calculate the distances.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance between points.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      def initialize(min_samples: 10, min_cluster_size: nil, metric: 'euclidean')
        check_params_numeric(min_samples: min_samples)
        check_params_numeric_or_nil(min_cluster_size: min_cluster_size)
        check_params_string(metric: metric)
        check_params_positive(min_samples: min_samples)
        @params = {}
        @params[:min_samples] = min_samples
        @params[:min_cluster_size] = min_cluster_size || min_samples
        @params[:metric] = metric == 'precomputed' ? 'precomputed' : 'euclidean'
        @labels = nil
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> HDBSCAN
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [HDBSCAN] The learned cluster analyzer itself.
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
          @parent = Numo::Int32.new(n).seq
          @rank = Numo::Int32.zeros(n)
        end

        # @!visibility private
        def union(x, y)
          x_root = find(x)
          y_root = find(y)

          return if x_root == y_root

          # :nocov:
          if @rank[x_root] < @rank[y_root]
            @parent[x_root] = y_root
          else
            @parent[y_root] = x_root
            @rank[x_root] += 1 if @rank[x_root] == @rank[y_root]
          end
          # :nocov:

          nil
        end

        # @!visibility private
        def find(x)
          @parent[x] = find(@parent[x]) if @parent[x] != x
          @parent[x]
        end
      end

      # @!visibility private
      class Node
        # @!visibility private
        attr_reader :x, :y, :weight, :n_elements

        # @!visibility private
        def initialize(x:, y:, weight:, n_elements: 0)
          @x = x
          @y = y
          @weight = weight
          @n_elements = n_elements
        end

        # @!visibility private
        def ==(other)
          # :nocov:
          x == other.x && y == other.y && weight == other.weight && n_elements == other.n_elements
          # :nocov:
        end
      end

      private_constant :UnionFind, :Node

      def partial_fit(distance_mat)
        mr_distance_mat = mutual_reachability_distances(distance_mat, @params[:min_samples])
        hierarchy = Rumale::Clustering::SingleLinkage.new(n_clusters: 1, metric: 'precomputed').fit(mr_distance_mat).hierarchy
        tree = condense_tree(hierarchy, @params[:min_cluster_size])
        stabilities = cluster_stability(tree)
        flatten(tree, stabilities)
      end

      def mutual_reachability_distances(distance_mat, min_samples)
        core_distances = distance_mat.sort(axis: 1)[true, min_samples + 1]
        Numo::DFloat.maximum(core_distances.expand_dims(1), Numo::DFloat.maximum(core_distances, distance_mat))
      end

      def breadth_first_search_hierarchy(hierarchy, root)
        n_edges = hierarchy.size
        n_points = n_edges + 1
        to_process = [root]
        res = []
        while to_process.any?
          res.concat(to_process)
          to_process = to_process.select { |n| n >= n_points }.map { |n| n - n_points }
          to_process = to_process.map { |n| [hierarchy[n].x, hierarchy[n].y] }.flatten if to_process.any?
        end
        res
      end

      # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/MethodLength, Metrics/PerceivedComplexity
      def condense_tree(hierarchy, min_cluster_size)
        n_edges = hierarchy.size
        root = 2 * n_edges
        n_points = n_edges + 1
        next_label = n_points + 1

        node_ids = breadth_first_search_hierarchy(hierarchy, root)

        relabel = Numo::Int32.zeros(root + 1)
        relabel[root] = n_points
        res = []
        visited = {}

        node_ids.each do |n_id|
          next if visited[n_id] || n_id < n_points

          edge = hierarchy[n_id - n_points]

          density = edge.weight > 0.0 ? 1.fdiv(edge.weight) : Float::INFINITY
          n_x_elements = edge.x >= n_points ? hierarchy[edge.x - n_points].n_elements : 1
          n_y_elements = edge.y >= n_points ? hierarchy[edge.y - n_points].n_elements : 1

          if n_x_elements >= min_cluster_size && n_y_elements >= min_cluster_size
            relabel[edge.x] = next_label
            res.push(Node.new(x: relabel[n_id], y: relabel[edge.x], weight: density, n_elements: n_x_elements))
            next_label += 1
            relabel[edge.y] = next_label
            res.push(Node.new(x: relabel[n_id], y: relabel[edge.y], weight: density, n_elements: n_y_elements))
            next_label += 1
          elsif n_x_elements < min_cluster_size && n_y_elements < min_cluster_size
            breadth_first_search_hierarchy(hierarchy, edge.x).each do |sn_id|
              res.push(Node.new(x: relabel[n_id], y: sn_id, weight: density, n_elements: 1)) if sn_id < n_points
              visited[sn_id] = true
            end
            breadth_first_search_hierarchy(hierarchy, edge.y).each do |sn_id|
              res.push(Node.new(x: relabel[n_id], y: sn_id, weight: density, n_elements: 1)) if sn_id < n_points
              visited[sn_id] = true
            end
          elsif n_x_elements < min_cluster_size
            relabel[edge.y] = relabel[n_id]
            breadth_first_search_hierarchy(hierarchy, edge.x).each do |sn_id|
              res.push(Node.new(x: relabel[n_id], y: sn_id, weight: density, n_elements: 1)) if sn_id < n_points
              visited[sn_id] = true
            end
          elsif n_y_elements < min_cluster_size
            relabel[edge.x] = relabel[n_id]
            breadth_first_search_hierarchy(hierarchy, edge.y).each do |sn_id|
              res.push(Node.new(x: relabel[n_id], y: sn_id, weight: density, n_elements: 1)) if sn_id < n_points
              visited[sn_id] = true
            end
          end
        end
        res
      end

      def cluster_stability(tree)
        tree.sort! { |a, b| a.weight <=> b.weight }

        root = tree.map(&:x).min
        child_max = tree.map(&:y).max
        child_max = root if child_max < root
        densities = Numo::DFloat.zeros(child_max + 1) + Float::INFINITY

        current = tree[0].y
        density_min = tree[0].weight
        tree.each do |edge|
          if edge.x == current
            density_min = [density_min, edge.weight].min
          else
            densities[current] = density_min
            current = edge.y
            density_min = edge.weight
          end
        end

        densities[current] = density_min if current != tree[0].y
        densities[root] = 0.0

        tree.each_with_object({}) do |edge, stab|
          stab[edge.x] ||= 0.0
          stab[edge.x] += (edge.weight - densities[edge.x]) * edge.n_elements
        end
      end

      def breadth_first_search_tree(tree, root)
        to_process = [root]
        res = []
        while to_process.any?
          res.concat(to_process)
          to_process = tree.select { |v| to_process.include?(v.x) }.map(&:y)
        end
        res
      end

      def flatten(tree, stabilities)
        node_ids = stabilities.keys.sort.reverse.slice(0, stabilities.size - 1)

        cluster_tree = tree.select { |edge| edge.n_elements > 1 }
        is_cluster = node_ids.each_with_object({}) { |n_id, h| h[n_id] = true }

        node_ids.each do |n_id|
          children = cluster_tree.select { |node| node.x == n_id }.map(&:y)
          subtree_stability = children.inject(0.0) { |sum, c_id| sum + stabilities[c_id] }
          if subtree_stability > stabilities[n_id]
            is_cluster[n_id] = false
            stabilities[n_id] = subtree_stability
          else
            breadth_first_search_tree(cluster_tree, n_id).each do |sn_id|
              is_cluster[sn_id] = false if sn_id != n_id
            end
          end
        end

        cluster_label_map = {}
        is_cluster.select { |_k, v| v == true }.keys.uniq.sort.each_with_index { |n_idx, c_idx| cluster_label_map[n_idx] = c_idx }

        parent_arr = tree.map(&:x)
        uf = UnionFind.new(parent_arr.max + 1)
        tree.each { |edge| uf.union(edge.x, edge.y) if cluster_label_map[edge.y].nil? }

        root = parent_arr.min
        res = Numo::Int32.zeros(root)
        root.times do |n|
          cluster = uf.find(n)
          res[n] = cluster < root ? -1 : cluster_label_map[cluster] || -1
        end
        res
      end
      # rubocop:enable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/MethodLength, Metrics/PerceivedComplexity
    end
  end
end
