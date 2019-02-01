# frozen_string_literal: true

require 'svmkit/base/base_estimator'
require 'svmkit/base/regressor'
require 'svmkit/tree/node'

module SVMKit
  module Tree
    # DecisionTreeRegressor is a class that implements decision tree for regression.
    #
    # @example
    #   estimator =
    #     SVMKit::Tree::DecisionTreeRegressor.new(
    #       max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    class DecisionTreeRegressor
      include Base::BaseEstimator
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
        @params = {}
        @params[:criterion] = criterion
        @params[:max_depth] = max_depth
        @params[:max_leaf_nodes] = max_leaf_nodes
        @params[:min_samples_leaf] = min_samples_leaf
        @params[:max_features] = max_features
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @criterion = :mse
        @criterion = :mae if @params[:criterion] == 'mae'
        @tree = nil
        @feature_importances = nil
        @n_leaves = nil
        @leaf_values = nil
        @rng = Random.new(@params[:random_seed])
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
        single_target = y.shape[1].nil?
        y = y.expand_dims(1) if single_target
        n_samples, n_features = x.shape
        @params[:max_features] = n_features if @params[:max_features].nil?
        @params[:max_features] = [@params[:max_features], n_features].min
        build_tree(x, y)
        @leaf_values = @leaf_values[true] if single_target
        eval_importance(n_samples, n_features)
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

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::Int32] (shape: [n_samples]) Leaf index for sample.
      def apply(x)
        check_sample_array(x)
        Numo::Int32[*(Array.new(x.shape[0]) { |n| apply_at_node(@tree, x[n, true]) })]
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about DecisionTreeRegressor
      def marshal_dump
        { params: @params,
          criterion: @criterion,
          tree: @tree,
          feature_importances: @feature_importances,
          leaf_values: @leaf_values,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @criterion = obj[:criterion]
        @tree = obj[:tree]
        @feature_importances = obj[:feature_importances]
        @leaf_values = obj[:leaf_values]
        @rng = obj[:rng]
        nil
      end

      private

      def apply_at_node(node, sample)
        return node.leaf_id if node.leaf
        return apply_at_node(node.left, sample) if node.right.nil?
        return apply_at_node(node.right, sample) if node.left.nil?
        if sample[node.feature_id] <= node.threshold
          apply_at_node(node.left, sample)
        else
          apply_at_node(node.right, sample)
        end
      end

      def build_tree(x, y)
        @n_leaves = 0
        @leaf_values = []
        @tree = grow_node(0, x, y, impurity(y))
        @leaf_values = Numo::DFloat.cast(@leaf_values)
        nil
      end

      def grow_node(depth, x, y, whole_impurity)
        unless @params[:max_leaf_nodes].nil?
          return nil if @n_leaves >= @params[:max_leaf_nodes]
        end

        n_samples, n_features = x.shape
        return nil if n_samples <= @params[:min_samples_leaf]

        node = Node.new(depth: depth, impurity: whole_impurity, n_samples: n_samples)

        return put_leaf(node, y) if (y - y.mean(0)).sum.abs.zero?

        unless @params[:max_depth].nil?
          return put_leaf(node, y) if depth == @params[:max_depth]
        end

        feature_id, threshold, left_ids, right_ids, left_impurity, right_impurity, gain =
          rand_ids(n_features).map { |f_id| [f_id, *best_split(x[true, f_id], y, whole_impurity)] }.max_by(&:last)

        return put_leaf(node, y) if gain.nil? || gain.zero?

        node.left = grow_node(depth + 1, x[left_ids, true], y[left_ids, true], left_impurity)
        node.right = grow_node(depth + 1, x[right_ids, true], y[right_ids, true], right_impurity)

        return put_leaf(node, y) if node.left.nil? && node.right.nil?

        node.feature_id = feature_id
        node.threshold = threshold
        node.leaf = false
        node
      end

      def put_leaf(node, values)
        node.probs = nil
        node.leaf = true
        node.leaf_id = @n_leaves
        @n_leaves += 1
        @leaf_values.push(values.mean(0))
        node
      end

      def rand_ids(n)
        [*0...n].sample(@params[:max_features], random: @rng)
      end

      def best_split(features, values, whole_impurity)
        n_samples = values.shape[0]
        features.to_a.uniq.sort.each_cons(2).map do |l, r|
          threshold = 0.5 * (l + r)
          left_ids = features.le(threshold).where
          right_ids = features.gt(threshold).where
          left_impurity = impurity(values[left_ids, true])
          right_impurity = impurity(values[right_ids, true])
          gain = whole_impurity -
                 left_impurity * left_ids.size.fdiv(n_samples) -
                 right_impurity * right_ids.size.fdiv(n_samples)
          [threshold, left_ids, right_ids, left_impurity, right_impurity, gain]
        end.max_by(&:last)
      end

      def impurity(values)
        send(@criterion, values)
      end

      def mse(values)
        ((values - values.mean(0))**2).mean
      end

      def mae(values)
        (values - values.mean(0)).abs.mean
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
