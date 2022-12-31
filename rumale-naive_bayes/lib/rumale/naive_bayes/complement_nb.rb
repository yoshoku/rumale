# frozen_string_literal: true

require 'rumale/naive_bayes/base_naive_bayes'

module Rumale
  module NaiveBayes
    # ComplementNB is a class that implements Complement Naive Bayes classifier.
    #
    # @example
    #   require 'rumale/naive_bayes/complement_nb'
    #
    #   estimator = Rumale::NaiveBayes::ComplementNB.new(smoothing_param: 1.0)
    #   estimator.fit(training_samples, training_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Rennie, J. D. M., Shih, L., Teevan, J., and Karger, D. R., "Tackling the Poor Assumptions of Naive Bayes Text Classifiers," ICML' 03, pp. 616--623, 2013.
    class ComplementNB < BaseNaiveBayes
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
      # @param norm [Boolean] The flag indicating whether to normlize the weight vectors.
      def initialize(smoothing_param: 1.0, norm: false)
        super()
        @params = {
          smoothing_param: smoothing_param,
          norm: norm
        }
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
        @class_log_probs = Numo::NMath.log(@class_priors)
        compl_features = Numo::DFloat[*@classes.to_a.map { |l| x[y.ne(l).where, true].sum(axis: 0) }]
        compl_features += @params[:smoothing_param]
        n_classes = @classes.size
        @feature_probs = compl_features / compl_features.sum(axis: 1).reshape(n_classes, 1)
        feature_log_probs = Numo::NMath.log(@feature_probs)
        @weights = if normalize?
                     feature_log_probs / feature_log_probs.sum(axis: 1).reshape(n_classes, 1)
                   else
                     -feature_log_probs
                   end
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence scores per sample for each class.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        @class_log_probs + x.dot(@weights.transpose)
      end

      private

      def normalize?
        @params[:norm] == true
      end
    end
  end
end
