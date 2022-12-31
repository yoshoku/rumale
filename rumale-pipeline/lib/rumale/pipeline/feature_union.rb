# frozen_string_literal: true

require 'rumale/base/estimator'

module Rumale
  module Pipeline
    # FeatureUnion is a class that implements the function concatenating the multi-transformer results.
    #
    # @example
    #   require 'rumale/kernel_approximation/rbf'
    #   require 'rumale/decomposition/pca'
    #   require 'rumale/pipeline/feature_union'
    #
    #   fu = Rumale::Pipeline::FeatureUnion.new(
    #     transformers: {
    #       'rbf': Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_components: 96, random_seed: 1),
    #       'pca': Rumale::Decomposition::PCA.new(n_components: 32)
    #     }
    #   )
    #   fu.fit(training_samples, traininig_labels)
    #   results = fu.predict(testing_samples)
    #
    #   # > p results.shape[1]
    #   # > 128
    #
    class FeatureUnion < ::Rumale::Base::Estimator
      # Return the transformers
      # @return [Hash]
      attr_reader :transformers

      # Create a new feature union.
      #
      # @param transformers [Hash] List of transformers.  The order of transforms follows the insertion order of hash keys.
      def initialize(transformers:)
        super()
        @params = {}
        @transformers = transformers
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the transformers.
      # @param y [Numo::NArray/Nil] (shape: [n_samples, n_outputs]) The target values or labels to be used for fitting the transformers.
      # @return [FeatureUnion] The learned feature union itself.
      def fit(x, y = nil)
        @transformers.each { |_k, t| t.fit(x, y) }
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the transformers.
      # @param y [Numo::NArray/Nil] (shape: [n_samples, n_outputs]) The target values or labels to be used for fitting the transformers.
      # @return [Numo::DFloat] (shape: [n_samples, sum_n_components]) The transformed and concatenated data.
      def fit_transform(x, y = nil)
        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned transformers.
      # @return [Numo::DFloat] (shape: [n_samples, sum_n_components]) The transformed and concatenated data.
      def transform(x)
        z = @transformers.values.map { |t| t.transform(x) }
        Numo::NArray.hstack(z)
      end
    end
  end
end
