# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/tree/vr_tree_classifier'
require 'rumale/ensemble/random_forest_classifier'
require 'rumale/ensemble/value'

module Rumale
  module Ensemble
    # VRTreesClassifier is a class that implements variable-random (VR) trees for classification.
    #
    # @example
    #   require 'rumale/ensemble/vr_trees_classifier'
    #
    #   estimator =
    #     Rumale::Ensemble::VRTreesClassifier.new(
    #       n_estimators: 10, criterion: 'gini', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Liu, F. T., Ting, K. M., Yu, Y., and Zhou, Z. H., "Spectrum of Variable-Random Trees," Journal of Artificial Intelligence Research, vol. 32, pp. 355--384, 2008.
    class VRTreesClassifier < RandomForestClassifier
      # Return the set of estimators.
      # @return [Array<VRTreeClassifier>]
      attr_reader :estimators

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the importance for each feature.
      # @return [Numo::DFloat] (size: n_features)
      attr_reader :feature_importances

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with variable-random trees.
      #
      # @param n_estimators [Integer] The numeber of trees for contructing variable-random trees.
      # @param criterion [String] The function to evalue spliting point. Supported criteria are 'gini' and 'entropy'.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, variable-random tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on variable-random tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers 'Math.sqrt(n_features)' features.
      # @param n_jobs [Integer] The number of jobs for running the fit method in parallel.
      #   If nil is given, the method does not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(n_estimators: 10,
                     criterion: 'gini', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1,
                     max_features: nil, n_jobs: nil, random_seed: nil)
        super
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [VRTreesClassifier] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        # Initialize some variables.
        n_features = x.shape[1]
        @params[:max_features] = Math.sqrt(n_features).to_i if @params[:max_features].nil?
        @params[:max_features] = @params[:max_features].clamp(1, n_features)
        @classes = Numo::Int32.asarray(y.to_a.uniq.sort)
        sub_rng = @rng.dup
        # Construct trees.
        rng_seeds = Array.new(@params[:n_estimators]) { sub_rng.rand(::Rumale::Ensemble::Value::SEED_BASE) }
        alpha_ratio = 0.5 / @params[:n_estimators]
        alphas = Array.new(@params[:n_estimators]) { |v| v * alpha_ratio }
        @estimators = if enable_parallel?
                        parallel_map(@params[:n_estimators]) { |n| plant_tree(alphas[n], rng_seeds[n]).fit(x, y) }
                      else
                        Array.new(@params[:n_estimators]) { |n| plant_tree(alphas[n], rng_seeds[n]).fit(x, y) }
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

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        super
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        super
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples, n_estimators]) Leaf index for sample.
      def apply(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        super
      end

      private

      def plant_tree(alpha, rnd_seed)
        ::Rumale::Tree::VRTreeClassifier.new(
          criterion: @params[:criterion], alpha: alpha, max_depth: @params[:max_depth],
          max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
          max_features: @params[:max_features], random_seed: rnd_seed
        )
      end
    end
  end
end
