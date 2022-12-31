# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Transfrom categorical features to integer values.
    #
    # @example
    #   require 'rumale/preprocessing/ordinal_encoder'
    #
    #   encoder = Rumale::Preprocessing::OrdinalEncoder.new
    #   training_samples = [['left', 10], ['right', 15], ['right', 20]]
    #   training_samples = Numo::NArray.asarray(training_samples)
    #   encoder.fit(training_samples)
    #   p encoder.categories
    #   # [["left", "right"], [10, 15, 20]]
    #   testing_samples = [['left', 20], ['right', 10]]
    #   testing_samples = Numo::NArray.asarray(testing_samples)
    #   encoded = encoder.transform(testing_samples)
    #   p encoded
    #   # Numo::DFloat#shape=[2,2]
    #   # [[0, 2],
    #   #  [1, 0]]
    #   p encoder.inverse_transform(encoded)
    #   # Numo::RObject#shape=[2,2]
    #   # [["left", 20],
    #   #  ["right", 10]]
    class OrdinalEncoder < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Return the array consists of categorical value each feature.
      # @return [Array] (size: n_features)
      attr_reader :categories

      # Create a new encoder that transform categorical features to integer values.
      #
      # @param categories [Nil/Array] The category list for each feature.
      #   If nil is given, extracted categories from the training data by calling the fit method are used.
      def initialize(categories: nil)
        super()
        @categories = categories
      end

      # Fit encoder by extracting the category for each feature.
      #
      # @overload fit(x) -> OrdinalEncoder
      #
      # @param x [Numo::NArray] (shape: [n_samples, n_features]) The samples consisting of categorical features.
      # @return [LabelEncoder]
      def fit(x, _y = nil)
        raise ArgumentError, 'Expect sample matrix to be 2-D array' unless x.shape.size == 2

        n_features = x.shape[1]
        @categories = Array.new(n_features) { |n| x[true, n].to_a.uniq.sort }
        self
      end

      # Fit encoder, then return encoded categorical features to integer values.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::NArray] (shape: [n_samples, n_features]) The samples consisting of categorical features.
      # @return [Numo::DFloat] The encoded categorical features to integer values.
      def fit_transform(x, _y = nil)
        raise ArgumentError, 'Expect sample matrix to be 2-D array' unless x.shape.size == 2

        fit(x).transform(x)
      end

      # Encode categorical features.
      #
      # @param x [Numo::NArray] (shape: [n_samples, n_features]) The samples consisting of categorical features.
      # @return [Numo::DFloat] The encoded categorical features to integer values.
      def transform(x)
        raise ArgumentError, 'Expect sample matrix to be 2-D array' unless x.shape.size == 2

        n_features = x.shape[1]
        if n_features != @categories.size
          raise ArgumentError,
                'Expect the number of features and the number of categories to be equal'
        end

        transformed = Array.new(n_features) do |n|
          x[true, n].to_a.map { |v| @categories[n].index(v) }
        end

        Numo::DFloat.asarray(transformed.transpose)
      end

      # Decode values to categorical features.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples consisting of values transformed from categorical features.
      # @return [Numo::NArray] The decoded features.
      def inverse_transform(x)
        n_features = x.shape[1]
        if n_features != @categories.size
          raise ArgumentError,
                'Expect the number of features and the number of categories to be equal'
        end

        inv_transformed = Array.new(n_features) do |n|
          x[true, n].to_a.map { |i| @categories[n][i.to_i] }
        end

        Numo::NArray.asarray(inv_transformed.transpose)
      end
    end
  end
end
