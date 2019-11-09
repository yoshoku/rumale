# frozen_string_literal: true

require 'rumale/values'
require 'rumale/base/base_estimator'
require 'rumale/base/regressor'
require 'rumale/tree/decision_tree_regressor'

module Rumale
  module Ensemble
    # RandomForestRegressor is a class that implements random forest for regression
    #
    # @example
    #   estimator =
    #     Rumale::Ensemble::RandomForestRegressor.new(
    #       n_estimators: 10, criterion: 'mse', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    class RandomForestRegressor
      include Base::BaseEstimator
      include Base::Regressor

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
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(n_estimators: 10,
                     criterion: 'mse', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1,
                     max_features: nil, n_jobs: nil, random_seed: nil)
        check_params_type_or_nil(Integer, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                          max_features: max_features, n_jobs: n_jobs, random_seed: random_seed)
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
        @params[:n_jobs] = n_jobs
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
        x = check_convert_sample_array(x)
        y = check_convert_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        # Initialize some variables.
        n_samples, n_features = x.shape
        @params[:max_features] = Math.sqrt(n_features).to_i unless @params[:max_features].is_a?(Integer)
        @params[:max_features] = [[1, @params[:max_features]].max, n_features].min
        single_target = y.shape[1].nil?
        sub_rng = @rng.dup
        rngs = Array.new(@params[:n_estimators]) { Random.new(sub_rng.rand(Rumale::Values.int_max)) }
        # Construct forest.
        @estimators =
          if enable_parallel?
            # :nocov:
            parallel_map(@params[:n_estimators]) do |n|
              bootstrap_ids = Array.new(n_samples) { rngs[n].rand(0...n_samples) }
              tree = plant_tree(rngs[n].rand(Rumale::Values.int_max))
              tree.fit(x[bootstrap_ids, true], single_target ? y[bootstrap_ids] : y[bootstrap_ids, true])
            end
            # :nocov:
          else
            Array.new(@params[:n_estimators]) do |n|
              bootstrap_ids = Array.new(n_samples) { rngs[n].rand(0...n_samples) }
              tree = plant_tree(rngs[n].rand(Rumale::Values.int_max))
              tree.fit(x[bootstrap_ids, true], single_target ? y[bootstrap_ids] : y[bootstrap_ids, true])
            end
          end
        @feature_importances =
          if enable_parallel?
            parallel_map(@params[:n_estimators]) { |n| @estimators[n].feature_importances }.reduce(&:+)
          else
            @estimators.map(&:feature_importances).reduce(&:+)
          end
        @feature_importances /= @feature_importances.sum
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted value per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        if enable_parallel?
          parallel_map(@params[:n_estimators]) { |n| @estimators[n].predict(x) }.reduce(&:+) / @params[:n_estimators]
        else
          @estimators.map { |tree| tree.predict(x) }.reduce(&:+) / @params[:n_estimators]
        end
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to assign each leaf.
      # @return [Numo::Int32] (shape: [n_samples, n_estimators]) Leaf index for sample.
      def apply(x)
        x = check_convert_sample_array(x)
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

      def plant_tree(rnd_seed)
        Tree::DecisionTreeRegressor.new(
          criterion: @params[:criterion], max_depth: @params[:max_depth],
          max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
          max_features: @params[:max_features], random_seed: rnd_seed
        )
      end
    end
  end
end
