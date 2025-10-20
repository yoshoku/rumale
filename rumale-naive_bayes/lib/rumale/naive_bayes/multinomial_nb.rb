# frozen_string_literal: true

require 'rumale/naive_bayes/base_naive_bayes'

module Rumale
  module NaiveBayes
    # MultinomialNB is a class that implements Multinomial Naive Bayes classifier.
    #
    # @example
    #   require 'rumale/naive_bayes/multinomial_nb'
    #
    #   estimator = Rumale::NaiveBayes::MultinomialNB.new(smoothing_param: 1.0)
    #   estimator.fit(training_samples, training_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Manning, C D., Raghavan, P., and Schutze, H., "Introduction to Information Retrieval," Cambridge University Press., 2008.
    class MultinomialNB < BaseNaiveBayes
      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the prior probabilities of the classes.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :class_priors

      # Return the conditional probabilities for features of each class.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :feature_probs

      # Create a new classifier with Multinomial Naive Bayes.
      #
      # @param smoothing_param [Float] The Laplace smoothing parameter.
      def initialize(smoothing_param: 1.0)
        super()
        @params = { smoothing_param: smoothing_param }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The categorical variables (e.g. labels)
      #   to be used for fitting the model.
      # @return [MultinomialNB] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        n_samples, = x.shape
        @classes = Numo::Int32[*y.to_a.uniq.sort]
        @class_priors = Numo::DFloat[*@classes.to_a.map { |l| y.eq(l).count / n_samples.to_f }]
        count_features = Numo::DFloat[*@classes.to_a.map { |l| x[y.eq(l).where, true].sum(axis: 0) }]
        count_features += @params[:smoothing_param]
        n_classes = @classes.size
        @feature_probs = count_features / count_features.sum(axis: 1).reshape(n_classes, 1)
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence scores per sample for each class.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_classes = @classes.size
        bin_x = Numo::DFloat.cast(x.gt(0))
        log_likelihoods = Array.new(n_classes) do |l|
          Math.log(@class_priors[l]) + (bin_x * Numo::NMath.log(@feature_probs[l, true])).sum(axis: 1)
        end
        Numo::DFloat[*log_likelihoods].transpose.dup
      end
    end
  end
end
