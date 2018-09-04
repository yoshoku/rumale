# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'
require 'svmkit/tree/node'

module SVMKit
  # This module consists of the classes that implement tree models.
  module Tree
    # DecisionTreeClassifier is a class that implements decision tree for classification.
    #
    # @example
    #   estimator =
    #     SVMKit::Tree::DecisionTreeClassifier.new(
    #       criterion: 'gini', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    class DecisionTreeClassifier
      include Base::BaseEstimator
      include Base::Classifier

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
      # @param criterion [String] The function to evalue spliting point. Supported criteria are 'gini' and 'entropy'.
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
        SVMKit::Validation.check_params_type_or_nil(Integer, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                                             max_features: max_features, random_seed: random_seed)
        SVMKit::Validation.check_params_integer(min_samples_leaf: min_samples_leaf)
        SVMKit::Validation.check_params_string(criterion: criterion)
        SVMKit::Validation.check_params_positive(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                                 min_samples_leaf: min_samples_leaf, max_features: max_features)
        @params = {}
        @params[:criterion] = criterion
        @params[:max_depth] = max_depth
        @params[:max_leaf_nodes] = max_leaf_nodes
        @params[:min_samples_leaf] = min_samples_leaf
        @params[:max_features] = max_features
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @criterion = :gini
        @criterion = :entropy if @params[:criterion] == 'entropy'
        @tree = nil
        @classes = nil
        @feature_importances = nil
        @n_leaves = nil
        @leaf_labels = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [DecisionTreeClassifier] The learned classifier itself.
      def fit(x, y)
        SVMKit::Validation.check_sample_array(x)
        SVMKit::Validation.check_label_array(y)
        SVMKit::Validation.check_sample_label_size(x, y)
        n_samples, n_features = x.shape
        @params[:max_features] = n_features if @params[:max_features].nil?
        @params[:max_features] = [@params[:max_features], n_features].min
        @classes = Numo::Int32.asarray(y.to_a.uniq.sort)
        build_tree(x, y)
        eval_importance(n_samples, n_features)
        self
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        SVMKit::Validation.check_sample_array(x)
        @leaf_labels[apply(x)]
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        SVMKit::Validation.check_sample_array(x)
        Numo::DFloat[*(Array.new(x.shape[0]) { |n| predict_at_node(@tree, x[n, true]) })]
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Leaf index for sample.
      def apply(x)
        SVMKit::Validation.check_sample_array(x)
        Numo::Int32[*(Array.new(x.shape[0]) { |n| apply_at_node(@tree, x[n, true]) })]
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about DecisionTreeClassifier
      def marshal_dump
        { params: @params,
          classes: @classes,
          criterion: @criterion,
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
        @criterion = obj[:criterion]
        @tree = obj[:tree]
        @feature_importances = obj[:feature_importances]
        @leaf_labels = obj[:leaf_labels]
        @rng = obj[:rng]
        nil
      end

      private

      def predict_at_node(node, sample)
        return node.probs if node.leaf
        branch_at_node('predict', node, sample)
      end

      def apply_at_node(node, sample)
        return node.leaf_id if node.leaf
        branch_at_node('apply', node, sample)
      end

      def branch_at_node(action, node, sample)
        return send("#{action}_at_node", node.left, sample) if node.right.nil?
        return send("#{action}_at_node", node.right, sample) if node.left.nil?
        if sample[node.feature_id] <= node.threshold
          send("#{action}_at_node", node.left, sample)
        else
          send("#{action}_at_node", node.right, sample)
        end
      end

      def build_tree(x, y)
        @n_leaves = 0
        @leaf_labels = []
        @tree = grow_node(0, x, y)
        @leaf_labels = Numo::Int32[*@leaf_labels]
        nil
      end

      def grow_node(depth, x, y)
        if @params[:max_leaf_nodes].is_a?(Integer)
          return nil if @n_leaves >= @params[:max_leaf_nodes]
        end

        n_samples, n_features = x.shape
        if @params[:min_samples_leaf].is_a?(Integer)
          return nil if n_samples <= @params[:min_samples_leaf]
        end

        node = Node.new(depth: depth, impurity: impurity(y), n_samples: n_samples)

        return put_leaf(node, y) if y.to_a.uniq.size == 1

        if @params[:max_depth].is_a?(Integer)
          return put_leaf(node, y) if depth == @params[:max_depth]
        end

        feature_id, threshold, left_ids, right_ids, max_gain =
          rand_ids(n_features).map { |f_id| [f_id, *best_split(x[true, f_id], y)] }.max_by(&:last)
        return put_leaf(node, y) if max_gain.nil?
        return put_leaf(node, y) if max_gain.zero?

        node.left = grow_node(depth + 1, x[left_ids, true], y[left_ids])
        node.right = grow_node(depth + 1, x[right_ids, true], y[right_ids])
        return put_leaf(node, y) if node.left.nil? && node.right.nil?

        node.feature_id = feature_id
        node.threshold = threshold
        node.leaf = false
        node
      end

      def put_leaf(node, y)
        node.probs = Numo::DFloat.cast(@classes.map { |c| y.eq(c).count_true }) / node.n_samples
        node.leaf = true
        node.leaf_id = @n_leaves
        @n_leaves += 1
        @leaf_labels.push(@classes[node.probs.max_index])
        node
      end

      def rand_ids(n)
        [*0...n].sample(@params[:max_features], random: @rng)
      end

      def best_split(features, labels)
        features.to_a.uniq.sort.each_cons(2).map do |l, r|
          threshold = 0.5 * (l + r)
          left_ids, right_ids = splited_ids(features, threshold)
          [threshold, left_ids, right_ids, gain(labels, labels[left_ids], labels[right_ids])]
        end.max_by(&:last)
      end

      def splited_ids(features, threshold)
        [features.le(threshold).where, features.gt(threshold).where]
      end

      def gain(labels, labels_left, labels_right)
        prob_left = labels_left.size.fdiv(labels.size)
        prob_right = labels_right.size.fdiv(labels.size)
        impurity(labels) - prob_left * impurity(labels_left) - prob_right * impurity(labels_right)
      end

      def impurity(labels)
        cls = labels.to_a.uniq.sort
        cls.size == 1 ? 0.0 : send(@criterion, Numo::DFloat[*(cls.map { |c| labels.eq(c).count_true.fdiv(labels.size) })])
      end

      def gini(posterior_probs)
        1.0 - (posterior_probs * posterior_probs).sum
      end

      def entropy(posterior_probs)
        -(posterior_probs * Numo::NMath.log(posterior_probs)).sum
      end

      def eval_importance(n_samples, n_features)
        @feature_importances = Numo::DFloat.zeros(n_features)
        eval_importance_at_node(@tree)
        @feature_importances /= n_samples
        normalizer = @feature_importances.sum
        @feature_importances /= normalizer if normalizer > 0.0
        nil
      end

      def eval_importance_at_node(node)
        return nil if node.leaf
        return nil if node.left.nil? || node.right.nil?
        gain = node.n_samples * node.impurity -
               node.left.n_samples * node.left.impurity - node.right.n_samples * node.right.impurity
        @feature_importances[node.feature_id] += gain
        eval_importance_at_node(node.left)
        eval_importance_at_node(node.right)
      end
    end
  end
end
