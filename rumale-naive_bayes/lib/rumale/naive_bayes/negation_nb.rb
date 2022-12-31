# frozen_string_literal: true

require 'rumale/naive_bayes/base_naive_bayes'

module Rumale
  module NaiveBayes
    # NegationNB is a class that implements Negation Naive Bayes classifier.
    #
    # @example
    #   require 'rumale/naive_bayes/negation_nb'
    #
    #   estimator = Rumale::NaiveBayes::NegationNB.new(smoothing_param: 1.0)
    #   estimator.fit(training_samples, training_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Komiya, K., Sato, N., Fujimoto, K., and Kotani, Y., "Negation Naive Bayes for Categorization of Product Pages on the Web," RANLP' 11, pp. 586--592, 2011.
    class NegationNB < BaseNaiveBayes
      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the prior probabilities of the classes.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :class_priors

      # Return the conditional probabilities for features of each class.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :feature_probs

      # Create a new classifier with Complement Naive Bayes.
      #
      # @param smoothing_param [Float] The smoothing parameter.
      def initialize(smoothing_param: 1.0)
        super()
        @params = { smoothing_param: smoothing_param }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The categorical variables (e.g. labels)
      #   to be used for fitting the model.
      # @return [ComplementNB] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        n_samples, = x.shape
        @classes = Numo::Int32[*y.to_a.uniq.sort]
        @class_priors = Numo::DFloat[*@classes.to_a.map { |l| y.eq(l).count.fdiv(n_samples) }]
        @class_log_probs = Numo::NMath.log(1 / (1 - @class_priors))
        compl_features = Numo::DFloat[*@classes.to_a.map { |l| x[y.ne(l).where, true].sum(axis: 0) }]
        compl_features += @params[:smoothing_param]
        n_classes = @classes.size
        @feature_probs = compl_features / compl_features.sum(axis: 1).reshape(n_classes, 1)
        @weights = Numo::NMath.log(@feature_probs)
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence scores per sample for each class.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        @class_log_probs - x.dot(@weights.transpose)
      end
    end
  end
end
