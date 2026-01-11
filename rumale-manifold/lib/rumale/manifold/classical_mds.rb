# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/utils'
require 'rumale/validation'
require 'rumale/pairwise_metric'

module Rumale
  module Manifold
    # ClassicalMDS is a class that implements classical multi-dimensional scaling.
    #
    # @example
    #   require 'rumale/manifold/classical_mds'
    #
    #   mds = Rumale::Manifold::ClassicalMDS.new(n_components: 2)
    #   representations = mds.fit_transform(data)
    #
    class ClassicalMDS < Rumale::Base::Estimator
      include Rumale::Base::Transformer

      # Return the data in representation space.
      # @return [Numo::DFloat] (shape: [n_samples, n_components])
      attr_reader :embedding

      # Create a new transformer with Classical MDS.
      #
      # @param n_components [Integer] The number of dimensions on representation space.
      # @param metric [String] The metric to calculate the distances in original space.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance in original space.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      def initialize(n_components: 2, metric: 'euclidean')
        super()
        @params = {
          n_components: n_components,
          metric: metric
        }
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> ClassicalMDS
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #     If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      #   @return [ClassicalMDS] The learned transformer itself.
      def fit(x, _not_used = nil)
        raise 'ClassicalMDS#fit requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        x = ::Rumale::Validation.check_convert_sample_array(x)
        if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
          raise ArgumentError, 'Expect the input distance matrix to be square.'
        end

        n_samples = x.shape[0]
        distance_mat = @params[:metric] == 'precomputed' ? x : ::Rumale::PairwiseMetric.euclidean_distance(x)

        centering_mat = Numo::DFloat.eye(n_samples) - Numo::DFloat.new(n_samples, n_samples).fill(1.fdiv(n_samples))
        kernel_mat = -0.5 * centering_mat.dot(distance_mat * distance_mat).dot(centering_mat)
        eig_vals, eig_vecs = Numo::Linalg.eigh(kernel_mat, vals_range: (n_samples - @params[:n_components])...n_samples)
        eig_vals = eig_vals.reverse
        eig_vecs = eig_vecs.reverse(1)
        @embedding = eig_vecs.dot(Numo::NMath.sqrt(eig_vals.abs).diag)

        @embedding = @embedding.flatten.dup if @params[:n_components] == 1

        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #     If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      #   @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _not_used = nil)
        raise 'ClassicalMDS#fit_transform requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        x = ::Rumale::Validation.check_convert_sample_array(x)
        fit(x)
        @embedding.dup
      end
    end
  end
end
