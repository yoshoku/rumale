# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/validation'
require 'rumale/tree/ext'
require 'rumale/tree/node'

module Rumale
  module Tree
    # GradientTreeRegressor is a class that implements decision tree for regression with exact gredy algorithm.
    # This class is used internally for estimators with gradient tree boosting.
    #
    # *Reference*
    # - Friedman, J H., "Greedy Function Approximation: A Gradient Boosting Machine," Annals of Statistics, 29 (5), pp. 1189--1232, 2001.
    # - Friedman, J H., "Stochastic Gradient Boosting," Computational Statistics and Data Analysis, 38 (4), pp. 367--378, 2002.
    # - Chen, T., and Guestrin, C., "XGBoost: A Scalable Tree Boosting System,"  Proc. KDD'16, pp. 785--794, 2016.
    class GradientTreeRegressor < ::Rumale::Base::Estimator
      include ::Rumale::Base::Regressor
      include ::Rumale::Tree::ExtGradientTreeRegressor

      # Return the importance for each feature.
      # The feature importances are calculated based on the numbers of times the feature is used for splitting.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :feature_importances

      # Return the learned tree.
      # @return [Node]
      attr_reader :tree

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Return the values assigned each leaf.
      # @return [Numo::DFloat] (shape: [n_leaves])
      attr_reader :leaf_weights

      # Initialize a gradient tree regressor
      #
      # @param reg_lambda [Float] The L2 regularization term on weight.
      # @param shrinkage_rate [Float] The shrinkage rate for weight.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, decision tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on decision tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers all features.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(reg_lambda: 0.0, shrinkage_rate: 1.0,
                     max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1, max_features: nil, random_seed: nil)
        super()
        @params = {
          reg_lambda: reg_lambda,
          shrinkage_rate: shrinkage_rate,
          max_depth: max_depth,
          max_leaf_nodes: max_leaf_nodes,
          min_samples_leaf: min_samples_leaf,
          max_features: max_features,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples]) The taget values to be used for fitting the model.
      # @param g [Numo::DFloat] (shape: [n_samples]) The gradient of loss function.
      # @param h [Numo::DFloat] (shape: [n_samples]) The hessian of loss function.
      # @return [GradientTreeRegressor] The learned regressor itself.
      def fit(x, y, g, h)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        # Initialize some variables.
        n_features = x.shape[1]
        @params[:max_features] ||= n_features
        @n_leaves = 0
        @leaf_weights = []
        @feature_importances = Numo::DFloat.zeros(n_features)
        @sub_rng = @rng.dup
        # Build tree.
        build_tree(x, y, g, h)
        @leaf_weights = Numo::DFloat[*@leaf_weights]
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (size: n_samples) Predicted values per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        @leaf_weights[apply(x)].dup
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Leaf index for sample.
      def apply(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        Numo::Int32[*Array.new(x.shape[0]) { |n| partial_apply(@tree, x[n, true]) }]
      end

      private

      def partial_apply(tree, sample)
        node = tree
        until node.leaf
          node = if node.right.nil?
                   node.left
                 elsif node.left.nil?
                   node.right
                 else
                   sample[node.feature_id] <= node.threshold ? node.left : node.right
                 end
        end
        node.leaf_id
      end

      def build_tree(x, y, g, h)
        @feature_ids = Array.new(x.shape[1]) { |v| v }
        @tree = grow_node(0, x, y, g, h)
        @feature_ids = nil
        nil
      end

      def grow_node(depth, x, y, g, h) # rubocop:disable Metrics/AbcSize
        # intialize some variables.
        sum_g = g.sum
        sum_h = h.sum
        n_samples = x.shape[0]
        node = Node.new(depth: depth, n_samples: n_samples)

        # terminate growing.
        return nil if !@params[:max_leaf_nodes].nil? && @n_leaves >= @params[:max_leaf_nodes]
        return nil if n_samples < @params[:min_samples_leaf]
        return put_leaf(node, sum_g, sum_h) if n_samples == @params[:min_samples_leaf]
        return put_leaf(node, sum_g, sum_h) if !@params[:max_depth].nil? && depth == @params[:max_depth]
        return put_leaf(node, sum_g, sum_h) if stop_growing?(y)

        # calculate optimal parameters.
        feature_id, threshold, gain = rand_ids.map { |n| [n, *best_split(x[true, n], g, h, sum_g, sum_h)] }.max_by(&:last)

        return put_leaf(node, sum_g, sum_h) if gain.nil? || gain.zero?

        left_ids = x[true, feature_id].le(threshold).where
        right_ids = x[true, feature_id].gt(threshold).where
        node.left = grow_node(depth + 1, x[left_ids, true], y[left_ids], g[left_ids], h[left_ids])
        node.right = grow_node(depth + 1, x[right_ids, true], y[right_ids], g[right_ids], h[right_ids])

        return put_leaf(node, sum_g, sum_h) if node.left.nil? && node.right.nil?

        @feature_importances[feature_id] += 1.0

        node.feature_id = feature_id
        node.threshold = threshold
        node.leaf = false
        node
      end

      def stop_growing?(y)
        y.to_a.uniq.size == 1
      end

      def put_leaf(node, sum_g, sum_h)
        node.probs = nil
        node.leaf = true
        node.leaf_id = @n_leaves
        weight = -@params[:shrinkage_rate] * sum_g / (sum_h + @params[:reg_lambda])
        @leaf_weights.push(weight)
        @n_leaves += 1
        node
      end

      def best_split(f, g, h, sum_g, sum_h)
        find_split_params(f.sort_index, f, g, h, sum_g, sum_h, @params[:reg_lambda])
      end

      def rand_ids
        @feature_ids.sample(@params[:max_features], random: @sub_rng)
      end
    end
  end
end
