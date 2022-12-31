# frozen_string_literal: true

require 'rumale/naive_bayes/base_naive_bayes'

module Rumale
  module NaiveBayes
    # GaussianNB is a class that implements Gaussian Naive Bayes classifier.
    #
    # @example
    #   require 'rumale/naive_bayes/gaussian_nb'
    #
    #   estimator = Rumale::NaiveBayes::GaussianNB.new
    #   estimator.fit(training_samples, training_labels)
    #   results = estimator.predict(testing_samples)
    class GaussianNB < BaseNaiveBayes
      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the prior probabilities of the classes.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :class_priors

      # Return the mean vectors of the classes.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :means

      # Return the variance vectors of the classes.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :variances

      # Create a new classifier with Gaussian Naive Bayes.
      def initialize
        super()
        @params = {}
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The categorical variables (e.g. labels)
      #   to be used for fitting the model.
      # @return [GaussianNB] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        n_samples, = x.shape
        @classes = Numo::Int32[*y.to_a.uniq.sort]
        @class_priors = Numo::DFloat[*@classes.to_a.map { |l| y.eq(l).count / n_samples.to_f }]
        @means = Numo::DFloat[*@classes.to_a.map { |l| x[y.eq(l).where, true].mean(0) }]
        @variances = Numo::DFloat[*@classes.to_a.map { |l| x[y.eq(l).where, true].var(0) }]
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence scores per sample for each class.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_classes = @classes.size
        log_likelihoods = Array.new(n_classes) do |l|
          Math.log(@class_priors[l]) - 0.5 * (
            Numo::NMath.log(2.0 * Math::PI * @variances[l, true]) +
            ((x - @means[l, true])**2 / @variances[l, true])).sum(axis: 1)
        end
        Numo::DFloat[*log_likelihoods].transpose.dup
      end
    end
  end
end
