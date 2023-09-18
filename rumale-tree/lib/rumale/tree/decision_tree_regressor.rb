# frozen_string_literal: true

require 'rumale/tree/base_decision_tree'
require 'rumale/base/regressor'

module Rumale
  module Tree
    # DecisionTreeRegressor is a class that implements decision tree for regression.
    #
    # @example
    #   require 'rumale/tree/decision_tree_regressor'
    #
    #   estimator =
    #     Rumale::Tree::DecisionTreeRegressor.new(
    #       max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    class DecisionTreeRegressor < BaseDecisionTree
      include ::Rumale::Base::Regressor
      include ::Rumale::Tree::ExtDecisionTreeRegressor

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

      # Create a new regressor with decision tree algorithm.
      #
      # @param criterion [String] The function to evaluate spliting point. Supported criteria are 'mae' and 'mse'.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, decision tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on decision tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers all features.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(criterion: 'mse', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1, max_features: nil,
                     random_seed: nil)
        super
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The taget values to be used for fitting the model.
      # @return [DecisionTreeRegressor] The learned regressor itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        n_samples, n_features = x.shape
        @params[:max_features] = n_features if @params[:max_features].nil?
        @params[:max_features] = [@params[:max_features], n_features].min
        @n_leaves = 0
        @leaf_values = []
        @feature_ids = Array.new(x.shape[1]) { |v| v }
        @sub_rng = @rng.dup
        build_tree(x, y)
        eval_importance(n_samples, n_features)
        @leaf_values = Numo::DFloat.cast(@leaf_values)
        @leaf_values = @leaf_values.flatten.dup if @leaf_values.shape[1] == 1
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        @leaf_values.shape[1].nil? ? @leaf_values[apply(x)].dup : @leaf_values[apply(x), true].dup
      end

      private

      def build_tree(x, y)
        y = y.expand_dims(1).dup if y.shape[1].nil?
        @tree = grow_node(0, x, y, impurity(y))
        nil
      end

      def stop_growing?(y)
        y.to_a.uniq.size == 1
      end

      def put_leaf(node, y)
        node.probs = nil
        node.leaf = true
        node.leaf_id = @n_leaves
        @n_leaves += 1
        @leaf_values.push(y.mean(0))
        node
      end

      def best_split(f, y, impurity)
        find_split_params(@params[:criterion], impurity, f.sort_index, f, y)
      end

      def impurity(y)
        node_impurity(@params[:criterion], y)
      end
    end
  end
end
