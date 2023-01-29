# frozen_string_literal: true

require 'rumale/utils'
require 'rumale/validation'
require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/tree/decision_tree_regressor'
require 'rumale/ensemble/value'

module Rumale
  module Ensemble
    # AdaBoostRegressor is a class that implements AdaBoost for regression.
    # This class uses decision tree for a weak learner.
    #
    # @example
    #   require 'rumale/ensemble/ada_boost_regressor'
    #
    #   estimator =
    #     Rumale::Ensemble::AdaBoostRegressor.new(
    #       n_estimators: 10, criterion: 'mse', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Shrestha, D. L., and Solomatine, D. P., "Experiments with AdaBoost.RT, an Improved Boosting Scheme for Regression," Neural Computation 18 (7), pp. 1678--1710, 2006.
    class AdaBoostRegressor < ::Rumale::Base::Estimator
      include ::Rumale::Base::Regressor

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
      # @param n_estimators [Integer] The numeber of decision trees for contructing AdaBoost regressor.
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
        super()
        @params = {
          n_estimators: n_estimators,
          threshold: threshold,
          exponent: exponent,
          criterion: criterion,
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
      # @param y [Numo::DFloat] (shape: [n_samples]) The target values to be used for fitting the model.
      # @return [AdaBoostRegressor] The learned regressor itself.
      def fit(x, y) # rubocop:disable Metrics/AbcSize, Metrics/MethodLength
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        unless y.ndim == 1
          raise ArgumentError,
                'AdaBoostRegressor supports only single-target variable regression; ' \
                'the target value array is expected to be 1-D'
        end

        # Initialize some variables.
        n_samples, n_features = x.shape
        @params[:max_features] = n_features unless @params[:max_features].is_a?(Integer)
        @params[:max_features] = [[1, @params[:max_features]].max, n_features].min # rubocop:disable Style/ComparableClamp
        observation_weights = Numo::DFloat.zeros(n_samples) + 1.fdiv(n_samples)
        @estimators = []
        @estimator_weights = []
        @feature_importances = Numo::DFloat.zeros(n_features)
        sub_rng = @rng.dup
        # Construct forest.
        @params[:n_estimators].times do |_t|
          # Fit weak learner.
          ids = ::Rumale::Utils.choice_ids(n_samples, observation_weights, sub_rng)
          tree = ::Rumale::Tree::DecisionTreeRegressor.new(
            criterion: @params[:criterion], max_depth: @params[:max_depth],
            max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
            max_features: @params[:max_features], random_seed: sub_rng.rand(::Rumale::Ensemble::Value::SEED_BASE)
          )
          tree.fit(x[ids, true], y[ids])
          pred = tree.predict(x)
          # Calculate errors.
          abs_err = ((pred - y) / y).abs
          sum_target = abs_err.gt(@params[:threshold])
          break if sum_target.count.zero?

          err = observation_weights[sum_target].sum
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
          update_target = abs_err.le(@params[:threshold])
          break if update_target.count.zero?

          update[update_target] = beta
          observation_weights *= update
          observation_weights = observation_weights.clip(1.0e-15, nil)
          sum_observation_weights = observation_weights.sum
          break if sum_observation_weights.zero?

          observation_weights /= sum_observation_weights
        end
        if @estimators.empty?
          warn('Failed to converge, check hyper-parameters of AdaBoostRegressor.')
          self
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
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_samples, = x.shape
        predictions = Numo::DFloat.zeros(n_samples)
        @estimators.size.times do |t|
          predictions += @estimator_weights[t] * @estimators[t].predict(x)
        end
        sum_weight = @estimator_weights.sum
        predictions / sum_weight
      end
    end
  end
end
