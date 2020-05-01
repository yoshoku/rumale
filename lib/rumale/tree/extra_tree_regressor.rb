# frozen_string_literal: true

require 'rumale/tree/decision_tree_regressor'

module Rumale
  module Tree
    # ExtraTreeRegressor is a class that implements extra randomized tree for regression.
    #
    # @example
    #   estimator =
    #     Rumale::Tree::ExtraTreeRegressor.new(
    #       max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Geurts, P., Ernst, D., and Wehenkel, L., "Extremely randomized trees," Machine Learning, vol. 63 (1), pp. 3--42, 2006.
    class ExtraTreeRegressor < DecisionTreeRegressor
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

      # Create a new regressor with extra randomized tree algorithm.
      #
      # @param criterion [String] The function to evaluate spliting point. Supported criteria are 'mae' and 'mse'.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, extra tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on extra tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers all features.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(criterion: 'mse', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1, max_features: nil,
                     random_seed: nil)
        check_params_numeric_or_nil(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                    max_features: max_features, random_seed: random_seed)
        check_params_numeric(min_samples_leaf: min_samples_leaf)
        check_params_string(criterion: criterion)
        check_params_positive(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                              min_samples_leaf: min_samples_leaf, max_features: max_features)
        super
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The taget values to be used for fitting the model.
      # @return [ExtraTreeRegressor] The learned regressor itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        super
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        super
      end

      private

      def best_split(features, y, whole_impurity)
        threshold = @sub_rng.rand(features.min..features.max)
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
