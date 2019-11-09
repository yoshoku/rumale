# frozen_string_literal: true

require 'rumale/rumale'
require 'rumale/tree/base_decision_tree'
require 'rumale/base/classifier'

module Rumale
  module Tree
    # DecisionTreeClassifier is a class that implements decision tree for classification.
    #
    # @example
    #   estimator =
    #     Rumale::Tree::DecisionTreeClassifier.new(
    #       criterion: 'gini', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    class DecisionTreeClassifier < BaseDecisionTree
      include Base::Classifier
      include ExtDecisionTreeClassifier

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

      # Create a new classifier with decision tree algorithm.
      #
      # @param criterion [String] The function to evaluate spliting point. Supported criteria are 'gini' and 'entropy'.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, decision tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on decision tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers all features.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(criterion: 'gini', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1, max_features: nil,
                     random_seed: nil)
        check_params_numeric_or_nil(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                    max_features: max_features, random_seed: random_seed)
        check_params_numeric(min_samples_leaf: min_samples_leaf)
        check_params_string(criterion: criterion)
        check_params_positive(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                              min_samples_leaf: min_samples_leaf, max_features: max_features)
        super
        @leaf_labels = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [DecisionTreeClassifier] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)
        n_samples, n_features = x.shape
        @params[:max_features] = n_features if @params[:max_features].nil?
        @params[:max_features] = [@params[:max_features], n_features].min
        uniq_y = y.to_a.uniq.sort
        @classes = Numo::Int32.asarray(uniq_y)
        @n_leaves = 0
        @leaf_labels = []
        @sub_rng = @rng.dup
        build_tree(x, y.map { |v| uniq_y.index(v) })
        eval_importance(n_samples, n_features)
        @leaf_labels = Numo::Int32[*@leaf_labels]
        self
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        @leaf_labels[apply(x)].dup
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = check_convert_sample_array(x)
        Numo::DFloat[*(Array.new(x.shape[0]) { |n| predict_proba_at_node(@tree, x[n, true]) })]
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about DecisionTreeClassifier
      def marshal_dump
        { params: @params,
          classes: @classes,
          tree: @tree,
          feature_importances: @feature_importances,
          leaf_labels: @leaf_labels,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @classes = obj[:classes]
        @tree = obj[:tree]
        @feature_importances = obj[:feature_importances]
        @leaf_labels = obj[:leaf_labels]
        @rng = obj[:rng]
        nil
      end

      private

      def predict_proba_at_node(node, sample)
        return node.probs if node.leaf
        return predict_proba_at_node(node.left, sample) if node.right.nil?
        return predict_proba_at_node(node.right, sample) if node.left.nil?
        if sample[node.feature_id] <= node.threshold
          predict_proba_at_node(node.left, sample)
        else
          predict_proba_at_node(node.right, sample)
        end
      end

      def stop_growing?(y)
        y[true, 0].to_a.uniq.size == 1
      end

      def put_leaf(node, y)
        node.probs = y.flatten.bincount(minlength: @classes.size) / node.n_samples.to_f
        node.leaf = true
        node.leaf_id = @n_leaves
        @n_leaves += 1
        @leaf_labels.push(@classes[node.probs.max_index])
        node
      end

      def best_split(features, y, whole_impurity)
        order = features.sort_index
        n_classes = @classes.size
        find_split_params(@params[:criterion], whole_impurity, order, features, y[true, 0], n_classes)
      end

      def impurity(y)
        n_elements = y.shape[0]
        n_classes = @classes.size
        node_impurity(@params[:criterion], y[true, 0].dup, n_elements, n_classes)
      end
    end
  end
end
