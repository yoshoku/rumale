# frozen_string_literal: true

require 'rumale/values'
require 'rumale/base/base_estimator'
require 'rumale/base/classifier'
require 'rumale/tree/decision_tree_classifier'

module Rumale
  # This module consists of the classes that implement ensemble-based methods.
  module Ensemble
    # RandomForestClassifier is a class that implements random forest for classification.
    #
    # @example
    #   estimator =
    #     Rumale::Ensemble::RandomForestClassifier.new(
    #       n_estimators: 10, criterion: 'gini', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    class RandomForestClassifier
      include Base::BaseEstimator
      include Base::Classifier

      # Return the set of estimators.
      # @return [Array<DecisionTreeClassifier>]
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

      # Create a new classifier with random forest.
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
      # @param n_jobs [Integer] The number of jobs for running the fit method in parallel.
      #   If nil is given, the method does not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(n_estimators: 10,
                     criterion: 'gini', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1,
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
        @classes = nil
        @feature_importances = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [RandomForestClassifier] The learned classifier itself.
      def fit(x, y)
        check_sample_array(x)
        check_label_array(y)
        check_sample_label_size(x, y)
        # Initialize some variables.
        n_samples, n_features = x.shape
        @params[:max_features] = Math.sqrt(n_features).to_i unless @params[:max_features].is_a?(Integer)
        @params[:max_features] = [[1, @params[:max_features]].max, n_features].min
        @classes = Numo::Int32.asarray(y.to_a.uniq.sort)
        # Construct forest.
        @estimators =
          if enable_parallel?
            rngs = Array.new(@params[:n_estimators]) { Random.new(@rng.rand(Rumale::Values.int_max)) }
            parallel_map(@params[:n_estimators]) do |n|
              bootstrap_ids = Array.new(n_samples) { rngs[n].rand(0...n_samples) }
              plant_tree(rngs[n].rand(Rumale::Values.int_max)).fit(x[bootstrap_ids, true], y[bootstrap_ids])
            end
          else
            Array.new(@params[:n_estimators]) do
              bootstrap_ids = Array.new(n_samples) { @rng.rand(0...n_samples) }
              plant_tree(@rng.rand(Rumale::Values.int_max)).fit(x[bootstrap_ids, true], y[bootstrap_ids])
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

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        check_sample_array(x)
        n_samples, = x.shape
        n_classes = @classes.size
        classes_arr = @classes.to_a
        ballot_box = Numo::DFloat.zeros(n_samples, n_classes)
        @estimators.each do |tree|
          predicted = tree.predict(x)
          n_samples.times do |n|
            class_id = classes_arr.index(predicted[n])
            ballot_box[n, class_id] += 1.0 unless class_id.nil?
          end
        end
        predicted = if enable_parallel?
                      parallel_map(n_samples) { |n| @classes[ballot_box[n, true].max_index] }
                    else
                      Array.new(n_samples) { |n| @classes[ballot_box[n, true].max_index] }
                    end
        Numo::Int32.asarray(predicted)
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        check_sample_array(x)
        n_samples, = x.shape
        n_classes = @classes.size
        classes_arr = @classes.to_a
        ballot_box = Numo::DFloat.zeros(n_samples, n_classes)
        @estimators.each do |tree|
          probs = tree.predict_proba(x)
          tree.classes.size.times do |n|
            class_id = classes_arr.index(tree.classes[n])
            ballot_box[true, class_id] += probs[true, n] unless class_id.nil?
          end
        end
        (ballot_box.transpose / ballot_box.sum(axis: 1)).transpose
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples, n_estimators]) Leaf index for sample.
      def apply(x)
        check_sample_array(x)
        Numo::Int32[*Array.new(@params[:n_estimators]) { |n| @estimators[n].apply(x) }].transpose
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about RandomForestClassifier.
      def marshal_dump
        { params: @params,
          estimators: @estimators,
          classes: @classes,
          feature_importances: @feature_importances,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @estimators = obj[:estimators]
        @classes = obj[:classes]
        @feature_importances = obj[:feature_importances]
        @rng = obj[:rng]
        nil
      end

      private

      def plant_tree(rnd_seed)
        Tree::DecisionTreeClassifier.new(
          criterion: @params[:criterion], max_depth: @params[:max_depth],
          max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
          max_features: @params[:max_features], random_seed: rnd_seed
        )
      end
    end
  end
end
