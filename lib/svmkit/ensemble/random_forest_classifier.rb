# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implement ensemble-based methods.
  module Ensemble
    # RandomForestClassifier is a class that implements random forest for classification.
    #
    # @example
    #   estimator =
    #     SVMKit::Ensemble::RandomForestClassifier.new(
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

      # Return the random generator for performing random sampling in the Pegasos algorithm.
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
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(n_estimators: 10, criterion: 'gini', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1,
                     max_features: nil, random_seed: nil)
        SVMKit::Validation.check_params_type_or_nil(Integer, max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                                             max_features: max_features, random_seed: random_seed)
        SVMKit::Validation.check_params_integer(n_estimators: n_estimators, min_samples_leaf: min_samples_leaf)
        SVMKit::Validation.check_params_string(criterion: criterion)
        SVMKit::Validation.check_params_positive(n_estimators: n_estimators, max_depth: max_depth,
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
        SVMKit::Validation.check_sample_array(x)
        SVMKit::Validation.check_label_array(y)
        SVMKit::Validation.check_sample_label_size(x, y)
        # Initialize some variables.
        n_samples, n_features = x.shape
        @params[:max_features] = n_features unless @params[:max_features].is_a?(Integer)
        @params[:max_features] = [[1, @params[:max_features]].max, Math.sqrt(n_features).to_i].min
        @classes = Numo::Int32.asarray(y.to_a.uniq.sort)
        # Construct forest.
        @estimators = Array.new(@params[:n_estimators]) do |_n|
          tree = Tree::DecisionTreeClassifier.new(
            criterion: @params[:criterion], max_depth: @params[:max_depth],
            max_leaf_nodes: @params[:max_leaf_nodes], min_samples_leaf: @params[:min_samples_leaf],
            max_features: @params[:max_features], random_seed: @params[:random_seed]
          )
          bootstrap_ids = Array.new(n_samples) { @rng.rand(0...n_samples) }
          tree.fit(x[bootstrap_ids, true], y[bootstrap_ids])
        end
        # Calculate feature importances.
        @feature_importances = Numo::DFloat.zeros(n_features)
        @estimators.each { |tree| @feature_importances += tree.feature_importances }
        @feature_importances /= @feature_importances.sum
        self
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        SVMKit::Validation.check_sample_array(x)
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
        Numo::Int32[*Array.new(n_samples) { |n| @classes[ballot_box[n, true].max_index] }]
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        SVMKit::Validation.check_sample_array(x)
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
        SVMKit::Validation.check_sample_array(x)
        Numo::Int32[*Array.new(@params[:n_estimators]) { |n| @estimators[n].apply(x) }].transpose
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about RandomForestClassifier
      def marshal_dump
        { params: @params, estimators: @estimators, classes: @classes,
          feature_importances: @feature_importances, rng: @rng }
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
    end
  end
end
