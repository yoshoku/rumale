# frozen_string_literal: true

require 'rumale/tree/decision_tree_regressor'

module Rumale
  module Tree
    # VRTreeRegressor is a class that implements Variable-Random (VR) tree for regression.
    #
    # @example
    #   require 'rumale/tree/vr_tree_regressor'
    #
    #   estimator =
    #     Rumale::Tree::VRTreeRegressor.new(
    #       max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Liu, F. T., Ting, K. M., Yu, Y., and Zhou, Z. H., "Spectrum of Variable-Random Trees," Journal of Artificial Intelligence Research, vol. 32, pp. 355--384, 2008.
    class VRTreeRegressor < DecisionTreeRegressor
      # Return the importance for each feature.
      # @return [Numo::DFloat] (size: n_features)
      attr_reader :feature_importances

      # Return the learned tree.
      # @return [Node]
      attr_reader :tree

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Return the values assigned each leaf.
      # @return [Numo::DFloat] (shape: [n_leafs, n_outputs])
      attr_reader :leaf_values

      # Create a new regressor with variable-random tree algorithm.
      #
      # @param criterion [String] The function to evaluate spliting point. Supported criteria are 'mae' and 'mse'.
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
      def initialize(criterion: 'mse', alpha: 0.5, max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1, max_features: nil,
                     random_seed: nil)
        super(criterion: criterion, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes, min_samples_leaf: min_samples_leaf,
              max_features: max_features, random_seed: random_seed)
        @params[:alpha] = alpha.clamp(0.0, 1.0)
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The taget values to be used for fitting the model.
      # @return [VRTreeRegressor] The learned regressor itself.

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.

      private

      def best_split(features, y, whole_impurity)
        r = -@sub_rng.rand(-1.0...0.0) # generate random number with (0, 1]
        return super if r <= @params[:alpha]

        fa, fb = features.to_a.uniq.sample(2, random: @sub_rng)
        threshold = 0.5 * (fa + fb)
        l_ids = features.le(threshold).where
        r_ids = features.gt(threshold).where
        l_impurity = l_ids.empty? ? 0.0 : impurity(y[l_ids, true])
        r_impurity = r_ids.empty? ? 0.0 : impurity(y[r_ids, true])
        gain = whole_impurity -
               l_impurity * l_ids.size.fdiv(y.shape[0]) -
               r_impurity * r_ids.size.fdiv(y.shape[0])
        [l_impurity, r_impurity, threshold, gain]
      end
    end
  end
end
