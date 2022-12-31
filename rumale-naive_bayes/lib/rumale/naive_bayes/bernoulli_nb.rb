# frozen_string_literal: true

require 'rumale/naive_bayes/base_naive_bayes'

module Rumale
  module NaiveBayes
    # BernoulliNB is a class that implements Bernoulli Naive Bayes classifier.
    #
    # @example
    #   require 'rumale/naive_bayes/bernoulli_nb'
    #
    #   estimator = Rumale::NaiveBayes::BernoulliNB.new(smoothing_param: 1.0, bin_threshold: 0.0)
    #   estimator.fit(training_samples, training_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Manning, C D., Raghavan, P., and Schutze, H., "Introduction to Information Retrieval," Cambridge University Press., 2008.
    class BernoulliNB < BaseNaiveBayes
      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the prior probabilities of the classes.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :class_priors

      # Return the conditional probabilities for features of each class.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :feature_probs

      # Create a new classifier with Bernoulli Naive Bayes.
      #
      # @param smoothing_param [Float] The Laplace smoothing parameter.
      # @param bin_threshold [Float] The threshold for binarizing of features.
      def initialize(smoothing_param: 1.0, bin_threshold: 0.0)
        super()
        @params = {
          smoothing_param: smoothing_param,
          bin_threshold: bin_threshold
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The categorical variables (e.g. labels)
      #   to be used for fitting the model.
      # @return [BernoulliNB] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        n_samples, = x.shape
        bin_x = Numo::DFloat[*x.gt(@params[:bin_threshold])]
        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_samples_each_class = Numo::DFloat[*@classes.to_a.map { |l| y.eq(l).count.to_f }]
        @class_priors = n_samples_each_class / n_samples
        count_features = Numo::DFloat[*@classes.to_a.map { |l| bin_x[y.eq(l).where, true].sum(axis: 0) }]
        count_features += @params[:smoothing_param]
        n_samples_each_class += 2.0 * @params[:smoothing_param]
        n_classes = @classes.size
        @feature_probs = count_features / n_samples_each_class.reshape(n_classes, 1)
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence scores per sample for each class.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_classes = @classes.size
        bin_x = Numo::DFloat[*x.gt(@params[:bin_threshold])]
        not_bin_x = Numo::DFloat[*x.le(@params[:bin_threshold])]
        log_likelihoods = Array.new(n_classes) do |l|
          Math.log(@class_priors[l]) + (
            (Numo::DFloat[*bin_x] * Numo::NMath.log(@feature_probs[l, true])).sum(axis: 1)
            (Numo::DFloat[*not_bin_x] * Numo::NMath.log(1.0 - @feature_probs[l, true])).sum(axis: 1))
        end
        Numo::DFloat[*log_likelihoods].transpose.dup
      end
    end
  end
end
