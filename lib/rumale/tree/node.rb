# frozen_string_literal: true

module Rumale
  module Tree
    # Node is a class that implements node used for construction of decision tree.
    # This class is used for internal data structures.
    class Node
      # @!visibility private
      attr_accessor :depth, :impurity, :n_samples, :probs, :leaf, :leaf_id, :left, :right, :feature_id, :threshold

      # Create a new node for decision tree.
      #
      # @param depth [Integer] The depth of the node in tree.
      # @param impurity [Float] The impurity of the node.
      # @param n_samples [Integer] The number of the samples in the node.
      # @param probs [Float] The probability of the node.
      # @param leaf [Boolean] The flag indicating whether the node is a leaf.
      # @param leaf_id [Integer] The leaf index of the node.
      # @param left [Node] The left node.
      # @param right [Node] The right node.
      # @param feature_id [Integer] The feature index used for evaluation.
      # @param threshold [Float] The threshold value of the feature for splitting the node.
      def initialize(depth: 0, impurity: 0.0, n_samples: 0, probs: 0.0,
                     leaf: false, leaf_id: nil,
                     left: nil, right: nil, feature_id: 0, threshold: 0.0)
        @depth = depth
        @impurity = impurity
        @n_samples = n_samples
        @probs = probs
        @leaf = leaf
        @leaf_id = leaf_id
        @left = left
        @right = right
        @feature_id = feature_id
        @threshold = threshold
      end
    end
  end
end
