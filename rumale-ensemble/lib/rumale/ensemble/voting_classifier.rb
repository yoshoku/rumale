# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/estimator'
require 'rumale/base/classifier'
require 'rumale/preprocessing/label_encoder'

module Rumale
  module Ensemble
    # VotingClassifier is a class that implements classifier with voting ensemble method.
    #
    # @example
    #   require 'rumale/ensemble/voting_classifier'
    #
    #   estimators = {
    #     lgr: Rumale::LinearModel::LogisticRegression.new(reg_param: 1e-2, random_seed: 1),
    #     mlp: Rumale::NeuralNetwork::MLPClassifier.new(hidden_units: [256], random_seed: 1),
    #     rnd: Rumale::Ensemble::RandomForestClassifier.new(random_seed: 1)
    #   }
    #   weights = { lgr: 0.2, mlp: 0.3, rnd: 0.5 }
    #
    #   classifier = Rumale::Ensemble::VotingClassifier.new(estimators: estimators, weights: weights, voting: 'soft')
    #   classifier.fit(x_train, y_train)
    #   results = classifier.predict(x_test)
    #
    # *Reference*
    # - Zhou, Z-H., "Ensemble Methods - Foundations and Algorithms," CRC Press Taylor and Francis Group, Chapman and Hall/CRC, 2012.
    class VotingClassifier < ::Rumale::Base::Estimator
      include ::Rumale::Base::Classifier

      # Return the sub-classifiers that voted.
      # @return [Hash<Symbol,Classifier>]
      attr_reader :estimators

      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Create a new ensembled classifier with voting rule.
      #
      # @param estimators [Hash<Symbol,Classifier>] The sub-classifiers to vote.
      # @param weights [Hash<Symbol,Float>] The weight value for each classifier.
      # @param voting [String] The voting rule for the predicted results of each classifier.
      #   If 'hard' is given, the ensembled classifier predicts the class label by majority vote.
      #   If 'soft' is given, the ensembled classifier uses the weighted average of predicted probabilities for the prediction.
      def initialize(estimators:, weights: nil, voting: 'hard')
        super()
        @estimators = estimators
        @params = {
          weights: weights || estimators.each_key.with_object({}) { |name, w| w[name] = 1.0 },
          voting: voting
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [VotingClassifier] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        @encoder = ::Rumale::Preprocessing::LabelEncoder.new
        y_encoded = @encoder.fit_transform(y)
        @classes = Numo::NArray[*@encoder.classes]
        @estimators.each_key { |name| @estimators[name].fit(x, y_encoded) }

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) The confidence score per sample.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        return predict_proba(x) if soft_voting?

        n_samples = x.shape[0]
        n_classes = @classes.size
        z = Numo::DFloat.zeros(n_samples, n_classes)
        @estimators.each do |name, estimator|
          estimator.predict(x).to_a.each_with_index { |c, i| z[i, c] += @params[:weights][name] }
        end
        z
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) The predicted class label per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        n_classes = @classes.size
        z = decision_function(x)
        predicted = z.max_index(axis: 1) - Numo::Int32.new(n_samples).seq * n_classes
        Numo::Int32.cast(@encoder.inverse_transform(predicted))
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probabilities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        n_classes = @classes.size
        z = Numo::DFloat.zeros(n_samples, n_classes)
        sum_weight = @params[:weights].each_value.sum
        @estimators.each do |name, estimator|
          z += @params[:weights][name] * estimator.predict_proba(x)
        end
        z /= sum_weight
      end

      private

      def soft_voting?
        @params[:voting] == 'soft'
      end
    end
  end
end
