# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/regressor'
require 'svmkit/tree/decision_tree_regressor'

module SVMKit
  module Ensemble
    # RandomForestRegressor is a class that implements random forest for regression
    #
    # @example
    #   estimator =
    #     SVMKit::Ensemble::RandomForestRegressor.new(
    #       n_estimators: 10, criterion: 'mse', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    class RandomForestRegressor
      include Base::BaseEstimator
      include Base::Regressor
      include Validation

      # Return the set of estimators.
      # @return [Array<DecisionTreeRegressor>]
      attr_reader :estimators

      # Return the importance for each feature.
      # @return [Numo::DFloat] (size: n_features)
      attr_reader :feature_importances

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with random forest.
      #
      # @param n_estimators [Integer] The numeber of decision trees for contructing random forest.
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
      def initialize(n_estimators: 10, criterion: 'mse', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1,
                     max_features: nil, random_seed: nil)
        check_params_type_or_nil(Integer, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                          max_features: max_features, random_seed: random_seed)
        check_params_integer(n_estimators: n_estimators, min_samples_leaf: min_samples_leaf)
        check_params_string(criterion: criterion)
        check_params_positive(n_estimators: n_estimators, max_depth: max_depth,
                              max_leaf_nodes: max_leaf_nodes, min_samples_leaf: min_samples_leaf,
                              max_features: max_features)
        @params = {}
        @params[:n_estimators] = n_estimators
        @params[:criterion] = criterion
        @params[:max_depth] = max_depth
        @params[:max_leaf_nodes] = max_leaf_nodes
        @params[:min_samples_leaf] = min_samples_leaf
        @params[:max_features] = max_features
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @estimators = nil
        @feature_importances = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [RandomForestRegressor] The learned regressor itself.
      def fit(x, y)
        check_sample_array(x)
        check_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        # Initialize some variables.
        n_samples, n_features = x.shape
        @params[:max_features] = Math.sqrt(n_features).to_i unless @params[:max_features].is_a?(Integer)
        @params[:max_features] = [[1, @params[:max_features]].max, n_features].min
        @feature_importances = Numo::DFloat.zeros(n_features)
        single_target = y.shape[1].nil?
        # Construct forest.
        @estimators = Array.new(@params[:n_estimators]) do
          tree = Tree::DecisionTreeRegressor.new(
            criterion: @params[:criterion], max_depth: @params[:max_depth],
            max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
            max_features: @params[:max_features], random_seed: @rng.rand(int_max)
          )
          bootstrap_ids = Array.new(n_samples) { @rng.rand(0...n_samples) }
          tree.fit(x[bootstrap_ids, true], single_target ? y[bootstrap_ids] : y[bootstrap_ids, true])
          @feature_importances += tree.feature_importances
          tree
        end
        @feature_importances /= @feature_importances.sum
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted value per sample.
      def predict(x)
        check_sample_array(x)
        @estimators.map { |est| est.predict(x) }.reduce(&:+) / @params[:n_estimators]
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to assign each leaf.
      # @return [Numo::Int32] (shape: [n_samples, n_estimators]) Leaf index for sample.
      def apply(x)
        SVMKit::Validation.check_sample_array(x)
        Numo::Int32[*Array.new(@params[:n_estimators]) { |n| @estimators[n].apply(x) }].transpose
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about RandomForestRegressor.
      def marshal_dump
        { params: @params,
          estimators: @estimators,
          feature_importances: @feature_importances,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @estimators = obj[:estimators]
        @feature_importances = obj[:feature_importances]
        @rng = obj[:rng]
        nil
      end

      private

      def int_max
        @int_max ||= 2**([42].pack('i').size * 16 - 2) - 1
      end
    end
  end
end
