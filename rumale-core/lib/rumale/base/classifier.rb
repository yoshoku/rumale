# frozen_string_literal: true

require 'numo/narray'

require 'rumale/validation'

module Rumale
  module Base
    # Module for all classifiers in Rumale.
    module Classifier
      # An abstract method for fitting a model.
      def fit
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # An abstract method for predicting labels.
      def predict
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # Calculate the mean accuracy of the given testing data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) Testing data.
      # @param y [Numo::Int32] (shape: [n_samples]) True labels for testing data.
      # @return [Float] Mean accuracy
      def score(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        predicted = predict(x)
        y.to_a.map.with_index { |label, n| label == predicted[n] ? 1 : 0 }.sum.fdiv(y.size)
      end
    end
  end
end
