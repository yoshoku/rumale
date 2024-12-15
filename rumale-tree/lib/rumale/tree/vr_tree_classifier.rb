# frozen_string_literal: true

require 'rumale/tree/decision_tree_classifier'

module Rumale
  module Tree
    # VRTreeClassifier is a class that implements Variable-Random (VR) tree for classification.
    #
    # @example
    #   require 'rumale/tree/vr_tree_classifier'
    #
    #   estimator =
    #     Rumale::Tree::VRTreeClassifier.new(
    #       criterion: 'gini', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Liu, F. T., Ting, K. M., Yu, Y., and Zhou, Z. H., "Spectrum of Variable-Random Trees," Journal of Artificial Intelligence Research, vol. 32, pp. 355--384, 2008.
    class VRTreeClassifier < DecisionTreeClassifier
      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the importance for each feature.
      # @return [Numo::DFloat] (size: n_features)
      attr_reader :feature_importances

      # Return the learned tree.
      # @return [Node]
      attr_reader :tree

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Return the labels assigned each leaf.
      # @return [Numo::Int32] (size: n_leafs)
      attr_reader :leaf_labels

      # Create a new classifier with variable-random tree algorithm.
      #
      # @param criterion [String] The function to evaluate spliting point. Supported criteria are 'gini' and 'entropy'.
      # @param alpha [Float] The probability of choosing a deterministic or random spliting point.
      #   If 1.0 is given, the tree is the same as the normal decision tree.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, variable-random tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on variable-random tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers all features.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(criterion: 'gini', alpha: 0.5, max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1, max_features: nil,
                     random_seed: nil)
        super(criterion: criterion, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes, min_samples_leaf: min_samples_leaf,
              max_features: max_features, random_seed: random_seed)
        @params[:alpha] = alpha.clamp(0.0, 1.0)
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [VRTreeClassifier] The learned classifier itself.

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.

      private

      def best_split(features, y, whole_impurity)
        r = -@sub_rng.rand(-1.0...0.0) # generate random number with (0, 1]
        return super if r <= @params[:alpha]

        fa, fb = features.to_a.uniq.sample(2, random: @sub_rng)
        threshold = 0.5 * (fa + fb)
        l_ids = features.le(threshold).where
        r_ids = features.gt(threshold).where
        l_impurity = l_ids.empty? ? 0.0 : impurity(y[l_ids])
        r_impurity = r_ids.empty? ? 0.0 : impurity(y[r_ids])
        gain = whole_impurity -
               l_impurity * l_ids.size.fdiv(y.size) -
               r_impurity * r_ids.size.fdiv(y.size)
        [l_impurity, r_impurity, threshold, gain]
      end
    end
  end
end
