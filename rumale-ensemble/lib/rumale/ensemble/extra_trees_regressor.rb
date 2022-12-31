# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/tree/extra_tree_regressor'
require 'rumale/ensemble/random_forest_regressor'
require 'rumale/ensemble/value'

module Rumale
  module Ensemble
    # ExtraTreesRegressor is a class that implements extremely randomized trees for regression
    # The algorithm of extremely randomized trees is similar to random forest.
    # The features of the algorithm of extremely randomized trees are
    # not to apply the bagging procedure and to randomly select the threshold for splitting feature space.
    #
    # @example
    #   @require 'rumale/ensemble/extra_trees_regressor'
    #
    #   estimator =
    #     Rumale::Ensemble::ExtraTreesRegressor.new(
    #       n_estimators: 10, criterion: 'mse', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Geurts, P., Ernst, D., and Wehenkel, L., "Extremely randomized trees," Machine Learning, vol. 63 (1), pp. 3--42, 2006.
    class ExtraTreesRegressor < RandomForestRegressor
      # Return the set of estimators.
      # @return [Array<ExtraTreeRegressor>]
      attr_reader :estimators

      # Return the importance for each feature.
      # @return [Numo::DFloat] (size: n_features)
      attr_reader :feature_importances

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with extremely randomized trees.
      #
      # @param n_estimators [Integer] The numeber of trees for contructing extremely randomized trees.
      # @param criterion [String] The function to evalue spliting point. Supported criteria are 'gini' and 'entropy'.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, extra tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on extra tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers 'Math.sqrt(n_features)' features.
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(n_estimators: 10,
                     criterion: 'mse', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1,
                     max_features: nil, n_jobs: nil, random_seed: nil)
        super
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [ExtraTreesRegressor] The learned regressor itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        # Initialize some variables.
        n_features = x.shape[1]
        @params[:max_features] = Math.sqrt(n_features).to_i if @params[:max_features].nil?
        @params[:max_features] = [[1, @params[:max_features]].max, n_features].min
        sub_rng = @rng.dup
        # Construct forest.
        rng_seeds = Array.new(@params[:n_estimators]) { sub_rng.rand(::Rumale::Ensemble::Value::SEED_BASE) }
        @estimators = if enable_parallel?
                        parallel_map(@params[:n_estimators]) { |n| plant_tree(rng_seeds[n]).fit(x, y) }
                      else
                        Array.new(@params[:n_estimators]) { |n| plant_tree(rng_seeds[n]).fit(x, y) }
                      end
        @feature_importances =
          if enable_parallel?
            parallel_map(@params[:n_estimators]) { |n| @estimators[n].feature_importances }.sum
          else
            @estimators.sum(&:feature_importances)
          end
        @feature_importances /= @feature_importances.sum
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted value per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        super
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to assign each leaf.
      # @return [Numo::Int32] (shape: [n_samples, n_estimators]) Leaf index for sample.
      def apply(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        super
      end

      private

      def plant_tree(rnd_seed)
        ::Rumale::Tree::ExtraTreeRegressor.new(
          criterion: @params[:criterion], max_depth: @params[:max_depth],
          max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
          max_features: @params[:max_features], random_seed: rnd_seed
        )
      end
    end
  end
end
