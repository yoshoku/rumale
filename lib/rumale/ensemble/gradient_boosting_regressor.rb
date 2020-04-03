# frozen_string_literal: true

require 'rumale/values'
require 'rumale/base/base_estimator'
require 'rumale/base/regressor'
require 'rumale/tree/gradient_tree_regressor'

module Rumale
  module Ensemble
    # GradientBoostingRegressor is a class that implements gradient tree boosting for regression.
    # The class use L2 loss for the loss function.
    #
    # @example
    #   estimator =
    #     Rumale::Ensemble::GradientBoostingRegressor.new(
    #       n_estimators: 100, learning_rate: 0.3, reg_lambda: 0.001, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *reference*
    # - J H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," Annals of Statistics, 29 (5), pp. 1189--1232, 2001.
    # - J H. Friedman, "Stochastic Gradient Boosting," Computational Statistics and Data Analysis, 38 (4), pp. 367--378, 2002.
    # - T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System,"  Proc. KDD'16, pp. 785--794, 2016.
    #
    class GradientBoostingRegressor
      include Base::BaseEstimator
      include Base::Regressor

      # Return the set of estimators.
      # @return [Array<GradientTreeRegressor>] or [Array<Array<GradientTreeRegressor>>]
      attr_reader :estimators

      # Return the importance for each feature.
      # The feature importances are calculated based on the numbers of times the feature is used for splitting.
      # @return [Numo::DFloat] (size: n_features)
      attr_reader :feature_importances

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with gradient tree boosting.
      #
      # @param n_estimators [Integer] The numeber of trees for contructing regressor.
      # @param learning_rate [Float] The boosting learining rate
      # @param reg_lambda [Float] The L2 regularization term on weight.
      # @param subsample [Float] The subsampling ratio of the training samples.
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
      def initialize(n_estimators: 100, learning_rate: 0.1, reg_lambda: 0.0, subsample: 1.0,
                     max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1,
                     max_features: nil, n_jobs: nil, random_seed: nil)
        check_params_numeric_or_nil(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                    max_features: max_features, n_jobs: n_jobs, random_seed: random_seed)
        check_params_numeric(n_estimators: n_estimators, min_samples_leaf: min_samples_leaf,
                             learning_rate: learning_rate, reg_lambda: reg_lambda, subsample: subsample)
        check_params_positive(n_estimators: n_estimators, learning_rate: learning_rate, reg_lambda: reg_lambda,
                              subsample: subsample, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                              min_samples_leaf: min_samples_leaf, max_features: max_features)
        @params = {}
        @params[:n_estimators] = n_estimators
        @params[:learning_rate] = learning_rate
        @params[:reg_lambda] = reg_lambda
        @params[:subsample] = subsample
        @params[:max_depth] = max_depth
        @params[:max_leaf_nodes] = max_leaf_nodes
        @params[:min_samples_leaf] = min_samples_leaf
        @params[:max_features] = max_features
        @params[:n_jobs] = n_jobs
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @estimators = nil
        @base_predictions = nil
        @feature_importances = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples]) The target values to be used for fitting the model.
      # @return [GradientBoostingRegressor] The learned regressor itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        # initialize some variables.
        n_features = x.shape[1]
        @params[:max_features] = n_features if @params[:max_features].nil?
        @params[:max_features] = [[1, @params[:max_features]].max, n_features].min
        n_outputs = y.shape[1].nil? ? 1 : y.shape[1]
        # train regressor.
        @base_predictions = n_outputs > 1 ? y.mean(0) : y.mean
        @estimators = if n_outputs > 1
                        multivar_estimators(x, y)
                      else
                        partial_fit(x, y, @base_predictions)
                      end
        # calculate feature importances.
        @feature_importances = if n_outputs > 1
                                 multivar_feature_importances
                               else
                                 @estimators.map(&:feature_importances).reduce(&:+)
                               end
        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples]) Predicted values per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        n_outputs = @estimators.first.is_a?(Array) ? @estimators.size : 1
        if n_outputs > 1
          multivar_predict(x)
        else
          if enable_parallel?
            parallel_map(@params[:n_estimators]) { |n| @estimators[n].predict(x) }.reduce(&:+) + @base_predictions
          else
            @estimators.map { |tree| tree.predict(x) }.reduce(&:+) + @base_predictions
          end
        end
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::Int32] (shape: [n_samples, n_estimators]) Leaf index for sample.
      def apply(x)
        x = check_convert_sample_array(x)
        n_outputs = @estimators.first.is_a?(Array) ? @estimators.size : 1
        leaf_ids = if n_outputs > 1
                     Array.new(n_outputs) { |n| @estimators[n].map { |tree| tree.apply(x) } }
                   else
                     @estimators.map { |tree| tree.apply(x) }
                   end
        Numo::Int32[*leaf_ids].transpose
      end

      private

      def partial_fit(x, y, init_pred)
        # initialize some variables.
        estimators = []
        n_samples = x.shape[0]
        n_sub_samples = [n_samples, [(n_samples * @params[:subsample]).to_i, 1].max].min
        whole_ids = Array.new(n_samples) { |v| v }
        y_pred = Numo::DFloat.ones(n_samples) * init_pred
        sub_rng = @rng.dup
        # grow trees.
        @params[:n_estimators].times do |_t|
          # subsampling
          ids = whole_ids.sample(n_sub_samples, random: sub_rng)
          x_sub = x[ids, true]
          y_sub = y[ids]
          y_pred_sub = y_pred[ids]
          # train tree
          g = gradient(y_sub, y_pred_sub)
          h = hessian(n_sub_samples)
          tree = plant_tree(sub_rng)
          tree.fit(x_sub, y_sub, g, h)
          estimators.push(tree)
          # update
          y_pred += tree.predict(x)
        end
        estimators
      end

      # for debug
      #
      # def loss(y_true, y_pred)
      #   ((y_true - y_pred)**2).mean
      # end

      def gradient(y_true, y_pred)
        y_pred - y_true
      end

      def hessian(n_samples)
        Numo::DFloat.ones(n_samples)
      end

      def plant_tree(sub_rng)
        Rumale::Tree::GradientTreeRegressor.new(
          reg_lambda: @params[:reg_lambda], shrinkage_rate: @params[:learning_rate],
          max_depth: @params[:max_depth],
          max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
          max_features: @params[:max_features], random_seed: sub_rng.rand(Rumale::Values.int_max)
        )
      end

      def multivar_estimators(x, y)
        n_outputs = y.shape[1]
        if enable_parallel?
          parallel_map(n_outputs) { |n| partial_fit(x, y[true, n], @base_predictions[n]) }
        else
          Array.new(n_outputs) { |n| partial_fit(x, y[true, n], @base_predictions[n]) }
        end
      end

      def multivar_feature_importances
        n_outputs = @estimators.size
        if enable_parallel?
          parallel_map(n_outputs) { |n| @estimators[n].map(&:feature_importances).reduce(&:+) }.reduce(&:+)
        else
          Array.new(n_outputs) { |n| @estimators[n].map(&:feature_importances).reduce(&:+) }.reduce(&:+)
        end
      end

      def multivar_predict(x)
        n_outputs = @estimators.size
        p = if enable_parallel?
              # :nocov:
              parallel_map(n_outputs) do |n|
                @estimators[n].map { |tree| tree.predict(x) }.reduce(&:+)
              end
              # :nocov:
            else
              Array.new(n_outputs) do |n|
                @estimators[n].map { |tree| tree.predict(x) }.reduce(&:+)
              end
            end
        Numo::DFloat.asarray(p).transpose + @base_predictions
      end
    end
  end
end
