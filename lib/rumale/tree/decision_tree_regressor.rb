# frozen_string_literal: true

require 'rumale/tree/base_decision_tree'
require 'rumale/base/regressor'

module Rumale
  module Tree
    # DecisionTreeRegressor is a class that implements decision tree for regression.
    #
    # @example
    #   estimator =
    #     Rumale::Tree::DecisionTreeRegressor.new(
    #       max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    class DecisionTreeRegressor < BaseDecisionTree
      include Base::Regressor

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
      # @param criterion [String] The function to evalue spliting point. Supported criteria are 'mae' and 'mse'.
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
        check_params_type_or_nil(Integer, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                          max_features: max_features, random_seed: random_seed)
        check_params_integer(min_samples_leaf: min_samples_leaf)
        check_params_string(criterion: criterion)
        check_params_positive(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                              min_samples_leaf: min_samples_leaf, max_features: max_features)
        super
        @leaf_values = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The taget values to be used for fitting the model.
      # @return [DecisionTreeRegressor] The learned regressor itself.
      def fit(x, y)
        check_sample_array(x)
        check_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        n_samples, n_features = x.shape
        @params[:max_features] = n_features if @params[:max_features].nil?
        @params[:max_features] = [@params[:max_features], n_features].min
        @n_leaves = 0
        @leaf_values = []
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
        check_sample_array(x)
        @leaf_values.shape[1].nil? ? @leaf_values[apply(x)] : @leaf_values[apply(x), true]
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about DecisionTreeRegressor
      def marshal_dump
        { params: @params,
          tree: @tree,
          feature_importances: @feature_importances,
          leaf_values: @leaf_values,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @tree = obj[:tree]
        @feature_importances = obj[:feature_importances]
        @leaf_values = obj[:leaf_values]
        @rng = obj[:rng]
        nil
      end

      private

      def stop_growing?(y)
        (y - y.mean(0)).sum.abs.zero?
      end

      def put_leaf(node, y)
        node.probs = nil
        node.leaf = true
        node.leaf_id = @n_leaves
        @n_leaves += 1
        @leaf_values.push(y.mean(0))
        node
      end

      def impurity(values)
        if @params[:criterion] == 'mae'
          (values - values.mean(0)).abs.mean
        else
          ((values - values.mean(0))**2).mean
        end
      end
    end
  end
end
