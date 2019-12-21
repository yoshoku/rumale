# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/pairwise_metric'
require 'rumale/base/base_estimator'

module Rumale
  module NearestNeighbors
    # VPTree is a class that implements the nearest neigbor searcher based on vantage point tree.
    # This implementation, unlike the paper, does not perform random sampling with vantage point selection.
    # This class is used internally for k-nearest neighbor estimators.
    #
    # *Reference*
    # P N. Yianilos, "Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces," Proc. SODA'93, pp. 311--321, 1993.
    class VPTree
      include Validation
      include Base::BaseEstimator

      # Return the training data.
      # @return [Numo::DFloat] (shape: [n_samples, n_features])
      attr_reader :data

      # Create a search index with vantage point tree algorithm.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to used generating search index.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      def initialize(x, min_samples_leaf: 1)
        check_params_numeric(min_samples_leaf: min_samples_leaf)
        check_params_positive(min_samples_leaf: min_samples_leaf)
        @params = {}
        @params[:min_samples_leaf] = min_samples_leaf
        @data = x
        @tree = build_tree(Numo::Int32.cast([*0...@data.shape[0]]))
      end

      # Search k-nearest neighbors of given query point.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      # @param k [Integer] The samples to be query points.
      # @return [Array<Array<Numo::Int32, Numo::DFloat>>] The indices and distances of retrieved k-nearest neighbors.
      def query(x, k = 1)
        x = check_convert_sample_array(x)
        check_params_numeric(k: k)
        check_params_positive(k: k)

        n_samples = x.shape[0]
        rel_ids = []
        rel_dists = []

        n_samples.times do |n|
          q = x[n, true]
          rel_node = search(q, @tree, k)
          dist_arr = calc_distances(q, @data[rel_node.sample_ids, true])
          rank_ids = dist_arr.sort_index[0...k]
          rel_ids.push(rel_node.sample_ids[rank_ids].dup)
          rel_dists.push(dist_arr[rank_ids].dup)
        end

        [Numo::Int32.cast(rel_ids), Numo::DFloat.cast(rel_dists)]
      end

      private

      Node = Struct.new(:sample_ids, :n_samples, :vantage_point_id, :threshold, :left, :right) do
        def leaf?
          vantage_point_id.nil?
        end
      end

      private_constant :Node

      def search(q, node, k, tau = Float::INFINITY)
        return node if node.leaf?

        dist = Math.sqrt(((q - @data[node.vantage_point_id, true])**2).sum)
        tau = dist if dist < tau

        # :nocov:
        if dist < node.threshold
          if dist - tau <= node.threshold
            node.left.n_samples < k ? node : search(q, node.left, k, tau)
          elsif dist + tau >= node.threshold
            node.right.n_samples < k ? node : search(q, node.right, k, tau)
          else
            node
          end
        else
          if dist + tau >= node.threshold
            node.right.n_samples < k ? node : search(q, node.right, k, tau)
          elsif dist - tau <= node.threshold
            node.left.n_samples < k ? node : search(q, node.left, k, tau)
          else
            node
          end
        end
        # :nocov:
      end

      def build_tree(sample_ids)
        n_samples = sample_ids.size
        node = Node.new
        node.n_samples = n_samples
        node.sample_ids = sample_ids
        return node if n_samples <= @params[:min_samples_leaf]

        vantage_point_id = select_vantage_point_id(sample_ids)
        distance_arr = calc_distances(@data[vantage_point_id, true], @data[sample_ids, true])
        threshold = distance_arr.median
        left_flgs = distance_arr.lt(threshold)
        right_flgs = distance_arr.ge(threshold)
        return node if left_flgs.count < @params[:min_samples_leaf] || right_flgs.count < @params[:min_samples_leaf]

        node.left = build_tree(sample_ids[left_flgs])
        node.right = build_tree(sample_ids[right_flgs])
        node.vantage_point_id = vantage_point_id
        node.threshold = threshold
        node
      end

      def select_vantage_point_id(sample_ids)
        dist_mat = Rumale::PairwiseMetric.euclidean_distance(@data[sample_ids, true])
        means = dist_mat.mean(0)
        vars = ((dist_mat - means)**2).mean(0)
        sample_ids[vars.max_index]
      end

      def calc_distances(q, x)
        Rumale::PairwiseMetric.euclidean_distance(q.expand_dims(0), x).flatten.dup
      end
    end
  end
end
