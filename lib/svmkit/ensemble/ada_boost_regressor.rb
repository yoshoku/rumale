# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/regressor'
require 'svmkit/tree/decision_tree_regressor'

module SVMKit
  module Ensemble
    # AdaBoostRegressor is a class that implements random forest for regression
    # This class uses decision tree for a weak learner.
    #
    # @example
    #   estimator =
    #     SVMKit::Ensemble::AdaBoostRegressor.new(
    #       n_estimators: 10, criterion: 'mse', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - D. L. Shrestha and D. P. Solomatine, "Experiments with AdaBoost.RT, an Improved Boosting Scheme for Regression," Neural Computation 18 (7), pp. 1678--1710, 2006.
    #
    class AdaBoostRegressor
      include Base::BaseEstimator
      include Base::Regressor
      include Validation

      # Return the set of estimators.
      # @return [Array<DecisionTreeRegressor>]
      attr_reader :estimators

      # Return the weight for each weak learner.
      # @return [Numo::DFloat] (size: n_estimates)
      attr_reader :estimator_weights

      # Return the importance for each feature.
      # @return [Numo::DFloat] (size: n_features)
      attr_reader :feature_importances

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with random forest.
      #
      # @param n_estimators [Integer] The numeber of decision trees for contructing random forest.
      # @param threshold [Float] The threshold for delimiting correct and incorrect predictions. That is constrained to [0, 1]
      # @param exponent [Float] The exponent for the weight of each weak learner.
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
      def initialize(n_estimators: 10, threshold: 0.2, exponent: 1.0,
                     criterion: 'mse', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1,
                     max_features: nil, random_seed: nil)
        check_params_type_or_nil(Integer, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                          max_features: max_features, random_seed: random_seed)
        check_params_integer(n_estimators: n_estimators, min_samples_leaf: min_samples_leaf)
        check_params_float(threshold: threshold, exponent: exponent)
        check_params_string(criterion: criterion)
        check_params_positive(n_estimators: n_estimators, threshold: threshold, exponent: exponent,
                              max_depth: max_depth,
                              max_leaf_nodes: max_leaf_nodes, min_samples_leaf: min_samples_leaf,
                              max_features: max_features)
        @params = {}
        @params[:n_estimators] = n_estimators
        @params[:threshold] = threshold
        @params[:exponent] = exponent
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
      # @param y [Numo::DFloat] (shape: [n_samples]) The target values to be used for fitting the model.
      # @return [AdaBoostRegressor] The learned regressor itself.
      def fit(x, y) # rubocop:disable Metrics/AbcSize
        check_sample_array(x)
        check_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        # Check target values
        raise ArgumentError, 'Expect target value vector to be 1-D arrray' unless y.shape.size == 1
        # Initialize some variables.
        n_samples, n_features = x.shape
        @params[:max_features] = n_features unless @params[:max_features].is_a?(Integer)
        @params[:max_features] = [[1, @params[:max_features]].max, n_features].min
        observation_weights = Numo::DFloat.zeros(n_samples) + 1.fdiv(n_samples)
        @estimators = []
        @estimator_weights = []
        @feature_importances = Numo::DFloat.zeros(n_features)
        # Construct forest.
        @params[:n_estimators].times do |_t|
          # Fit weak learner.
          ids = weighted_sampling(observation_weights)
          tree = Tree::DecisionTreeRegressor.new(
            criterion: @params[:criterion], max_depth: @params[:max_depth],
            max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
            max_features: @params[:max_features], random_seed: @rng.rand(int_max)
          )
          tree.fit(x[ids, true], y[ids])
          p = tree.predict(x)
          # Calculate errors.
          abs_err = ((p - y) / y).abs
          err = observation_weights[abs_err.gt(@params[:threshold])].sum
          break if err <= 0.0
          # Calculate weight.
          beta = err**@params[:exponent]
          weight = Math.log(1.fdiv(beta))
          # Store model.
          @estimators.push(tree)
          @estimator_weights.push(weight)
          @feature_importances += weight * tree.feature_importances
          # Update observation weights.
          update = Numo::DFloat.ones(n_samples)
          update[abs_err.le(@params[:threshold])] = beta
          observation_weights *= update
          observation_weights = observation_weights.clip(1.0e-15, nil)
          sum_observation_weights = observation_weights.sum
          break if sum_observation_weights.zero?
          observation_weights /= sum_observation_weights
        end
        @estimator_weights = Numo::DFloat.asarray(@estimator_weights)
        @feature_importances /= @estimator_weights.sum
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted value per sample.
      def predict(x)
        check_sample_array(x)
        n_samples, = x.shape
        predictions = Numo::DFloat.zeros(n_samples)
        @estimators.size.times do |t|
          predictions += @estimator_weights[t] * @estimators[t].predict(x)
        end
        sum_weight = @estimator_weights.sum
        predictions / sum_weight
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about AdaBoostRegressor.
      def marshal_dump
        { params: @params,
          estimators: @estimators,
          estimator_weights: @estimator_weights,
          feature_importances: @feature_importances,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @estimators = obj[:estimators]
        @estimator_weights = obj[:estimator_weights]
        @feature_importances = obj[:feature_importances]
        @rng = obj[:rng]
        nil
      end

      private

      def weighted_sampling(weights)
        Array.new(weights.size) do
          target = @rng.rand
          chosen = 0
          weights.each_with_index do |w, idx|
            if target <= w
              chosen = idx
              break
            end
            target -= w
          end
          chosen
        end
      end

      def int_max
        @int_max ||= 2**([42].pack('i').size * 16 - 2) - 1
      end
    end
  end
end
