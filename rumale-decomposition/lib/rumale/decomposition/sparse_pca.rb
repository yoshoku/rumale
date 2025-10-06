# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/utils'
require 'rumale/validation'

module Rumale
  module Decomposition
    # SparsePCA is a class that implements Sparse Principal Component Analysis.
    #
    # @example
    #   require 'numo/linalg'
    #   require 'rumale/decomposition/sparse_pca'
    #
    #   decomposer = Rumale::Decomposition::SparsePCA.new(n_components: 2, reg_param: 0.1)
    #   representaion = decomposer.fit_transform(samples)
    #   sparse_components = decomposer.components
    #
    # *Reference*
    # - Macky, L., "Deflation Methods for Sparse PCA," Advances in NIPS'08, pp. 1017--1024, 2008.
    # - Hein, M. and Bühler, T., "An Inverse Power Method for Nonlinear Eigenproblems with Applications in 1-Spectral Clustering and Sparse PCA," Advances in NIPS'10, pp. 847--855, 2010.
    class SparsePCA < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Returns the principal components.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Returns the mean vector.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :mean

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with Sparse PCA.
      #
      # @param n_components [Integer] The number of principal components.
      # @param reg_param [Float] The regularization parameter (interval: [0, 1]).
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: 2, reg_param: 0.001, max_iter: 1000, tol: 1e-6, random_seed: nil)
        super()

        warn('reg_param should be in the interval [0, 1].') unless (0..1).cover?(reg_param)

        @params = {
          n_components: n_components,
          reg_param: reg_param,
          max_iter: max_iter,
          tol: tol,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> SparsePCA
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [SparsePCA] The learned transformer itself.
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        # initialize some variables.
        @components = Numo::DFloat.zeros(@params[:n_components], x.shape[1])

        # centering.
        @mean = x.mean(axis: 0)
        centered_x = x - @mean

        # optimization.
        partial_fit(centered_x)

        @components = @components[0, true].dup if @params[:n_components] == 1

        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        (x - @mean).dot(@components.transpose)
      end

      private

      def partial_fit(x)
        sub_rng = @rng.dup
        n_samples, n_features = x.shape
        cov_mat = x.transpose.dot(x) / n_samples
        prj_mat = Numo::DFloat.eye(n_features)
        @params[:n_components].times do |i|
          f = ::Rumale::Utils.rand_normal(n_features, sub_rng)
          xf = x.dot(f)
          norm_xf = norm(xf, 2)
          coeff = coeff_numerator(f).fdiv(norm_xf)
          mu = cov_mat.dot(f) / norm_xf
          @params[:max_iter].times do |_t|
            g = sign(mu) * Numo::DFloat.maximum(coeff * mu.abs - @params[:reg_param], 0)
            f = g / norm(x.dot(g), 2)
            mu = cov_mat.dot(f) / norm(x.dot(f), 2)
            coeff_new = coeff_numerator(f)

            break if (coeff - coeff_new).abs.fdiv(coeff) < @params[:tol]

            coeff = coeff_new
          end

          # deflation
          q = prj_mat.dot(f)
          qqt = Numo::DFloat.eye(n_features) - q.outer(q)
          x = x.dot(qqt)
          cov_mat = qqt.dot(cov_mat).dot(qqt)
          prj_mat = prj_mat.dot(qqt)
          f /= norm(f, 2)

          @components[i, true] = f.dup
        end
      end

      def coeff_numerator(f)
        (1 - @params[:reg_param]) * norm(f, 2) + @params[:reg_param] * norm(f, 1)
      end

      def sign(v)
        r = Numo::DFloat.zeros(v.size)
        r[v.lt(0)] = -1
        r[v.gt(0)] = 1
        r
      end

      def norm(v, ord)
        nrm = if defined?(Numo::Linalg)
                Numo::Linalg.norm(v, ord)
              elsif ord == 2
                Math.sqrt(v.dot(v))
              else
                v.abs.sum
              end
        nrm.zero? ? 1.0 : nrm
      end
    end
  end
end
