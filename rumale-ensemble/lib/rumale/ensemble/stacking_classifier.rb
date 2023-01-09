# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/estimator'
require 'rumale/base/classifier'
require 'rumale/linear_model/logistic_regression'
require 'rumale/model_selection/stratified_k_fold'
require 'rumale/preprocessing/label_encoder'

module Rumale
  module Ensemble
    # StackingClassifier is a class that implements classifier with stacking method.
    #
    # @example
    #   require 'rumale/ensemble/stacking_classifier'
    #
    #   estimators = {
    #     lgr: Rumale::LinearModel::LogisticRegression.new(reg_param: 1e-2),
    #     mlp: Rumale::NeuralNetwork::MLPClassifier.new(hidden_units: [256], random_seed: 1),
    #     rnd: Rumale::Ensemble::RandomForestClassifier.new(random_seed: 1)
    #   }
    #   meta_estimator = Rumale::LinearModel::LogisticRegression.new
    #   classifier = Rumale::Ensemble::StackedClassifier.new(
    #     estimators: estimators, meta_estimator: meta_estimator, random_seed: 1
    #   )
    #   classifier.fit(training_samples, training_labels)
    #   results = classifier.predict(testing_samples)
    #
    # *Reference*
    # - Zhou, Z-H., "Ensemble Methods - Foundations and Algorithms," CRC Press Taylor and Francis Group, Chapman and Hall/CRC, 2012.
    class StackingClassifier < ::Rumale::Base::Estimator
      include ::Rumale::Base::Classifier

      # Return the base classifiers.
      # @return [Hash<Symbol,Classifier>]
      attr_reader :estimators

      # Return the meta classifier.
      # @return [Classifier]
      attr_reader :meta_estimator

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the method used by each base classifier.
      # @return [Hash<Symbol,Symbol>]
      attr_reader :stack_method

      # Create a new classifier with stacking method.
      #
      # @param estimators [Hash<Symbol,Classifier>] The base classifiers for extracting meta features.
      # @param meta_estimator [Classifier/Nil] The meta classifier that predicts class label.
      #   If nil is given, LogisticRegression is used.
      # @param n_splits [Integer] The number of folds for cross validation with stratified k-fold on meta feature extraction in training phase.
      # @param shuffle [Boolean] The flag indicating whether to shuffle the dataset on cross validation.
      # @param stack_method [String] The method name of base classifier for using meta feature extraction.
      #   If 'auto' is given, it searches the callable method in the order 'predict_proba', 'decision_function', and 'predict'
      #   on each classifier.
      # @param passthrough [Boolean] The flag indicating whether to concatenate the original features and meta features when training the meta classifier.
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator on cross validation.
      def initialize(estimators:, meta_estimator: nil, n_splits: 5, shuffle: true, stack_method: 'auto', passthrough: false,
                     random_seed: nil)
        super()
        @estimators = estimators
        @meta_estimator = meta_estimator || ::Rumale::LinearModel::LogisticRegression.new
        @params = {
          n_splits: n_splits,
          shuffle: shuffle,
          stack_method: stack_method,
          passthrough: passthrough,
          random_seed: random_seed || srand
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [StackedClassifier] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        n_samples, n_features = x.shape

        @encoder = ::Rumale::Preprocessing::LabelEncoder.new
        y_encoded = @encoder.fit_transform(y)
        @classes = Numo::NArray[*@encoder.classes]

        # training base classifiers with all training data.
        @estimators.each_key { |name| @estimators[name].fit(x, y_encoded) }

        # detecting feature extraction method and its size of output for each base classifier.
        @stack_method = detect_stack_method
        @output_size = detect_output_size(n_features)

        # extracting meta features with base classifiers.
        n_components = @output_size.values.sum
        z = Numo::DFloat.zeros(n_samples, n_components)

        kf = ::Rumale::ModelSelection::StratifiedKFold.new(
          n_splits: @params[:n_splits], shuffle: @params[:shuffle], random_seed: @params[:random_seed]
        )

        kf.split(x, y_encoded).each do |train_ids, valid_ids|
          x_train = x[train_ids, true]
          y_train = y_encoded[train_ids]
          x_valid = x[valid_ids, true]
          f_start = 0
          @estimators.each_key do |name|
            est_fold = Marshal.load(Marshal.dump(@estimators[name]))
            f_last = f_start + @output_size[name]
            f_position = @output_size[name] == 1 ? f_start : f_start...f_last
            z[valid_ids, f_position] = est_fold.fit(x_train, y_train).public_send(@stack_method[name], x_valid)
            f_start = f_last
          end
        end

        # concatenating original features.
        z = Numo::NArray.hstack([z, x]) if @params[:passthrough]

        # training meta classifier.
        @meta_estimator.fit(z, y_encoded)

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) The confidence score per sample.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        z = transform(x)
        @meta_estimator.decision_function(z)
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) The predicted class label per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        z = transform(x)
        Numo::Int32.cast(@encoder.inverse_transform(@meta_estimator.predict(z)))
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probabilities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) The predicted probability of each class per sample.
      def predict_proba(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        z = transform(x)
        @meta_estimator.predict_proba(z)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The meta features for samples.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        n_components = @output_size.values.sum
        z = Numo::DFloat.zeros(n_samples, n_components)
        f_start = 0
        @estimators.each_key do |name|
          f_last = f_start + @output_size[name]
          f_position = @output_size[name] == 1 ? f_start : f_start...f_last
          z[true, f_position] = @estimators[name].public_send(@stack_method[name], x)
          f_start = f_last
        end
        z = Numo::NArray.hstack([z, x]) if @params[:passthrough]
        z
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The meta features for training data.
      def fit_transform(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        fit(x, y).transform(x)
      end

      private

      STACK_METHODS = %i[predict_proba decision_function predict].freeze

      private_constant :STACK_METHODS

      def detect_stack_method
        if @params[:stack_method] == 'auto'
          @estimators.each_key.with_object({}) do |name, obj|
            obj[name] = STACK_METHODS.detect do |m|
              @estimators[name].respond_to?(m)
            end
          end
        else
          @estimators.each_key.with_object({}) { |name, obj| obj[name] = @params[:stack_method].to_sym }
        end
      end

      def detect_output_size(n_features)
        x_dummy = Numo::DFloat.new(2, n_features).rand
        @estimators.each_key.with_object({}) do |name, obj|
          output_dummy = @estimators[name].public_send(@stack_method[name], x_dummy)
          obj[name] = output_dummy.ndim == 1 ? 1 : output_dummy.shape[1]
        end
      end
    end
  end
end
