# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/transformer'

module SVMKit
  module Decomposition
    # NMF is a class that implements Non-negative Matrix Factorization.
    #
    # @example
    #   decomposer = SVMKit::Decomposition::NMF.new(n_components: 2)
    #   representaion = decomposer.fit_transform(samples)
    #
    # *Reference*
    # - W. Xu, X. Liu, and Y.Gong, "Document Clustering Based On Non-negative Matrix Factorization," Proc. SIGIR' 03 , pp. 267--273, 2003.
    class NMF
      include Base::BaseEstimator
      include Base::Transformer
      include Validation

      # Returns the factorization matrix.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with NMF.
      #
      # @param n_components [Integer] The number of components.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param eps [Float] A small value close to zero to avoid zero division error.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: 2, max_iter: 500, tol: 1.0e-4, eps: 1.0e-16, random_seed: nil)
        check_params_integer(n_components: n_components, max_iter: max_iter)
        check_params_float(tol: tol, eps: eps)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        check_params_positive(n_components: n_components, max_iter: max_iter, tol: tol, eps: eps)
        @params = {}
        @params[:n_components] = n_components
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:eps] = eps
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @components = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> NMF
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @return [NMF] The learned transformer itself.
      def fit(x, _y = nil)
        check_sample_array(x)
        partial_fit(x)
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _y = nil)
        check_sample_array(x)
        partial_fit(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        check_sample_array(x)
        partial_fit(x, false)
      end

      # Inverse transform the given transformed data with the learned model.
      #
      # @param z [Numo::DFloat] (shape: [n_samples, n_components]) The data to be restored into original space with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_featuress]) The restored data.
      def inverse_transform(z)
        check_sample_array(z)
        z.dot(@components)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data.
      def marshal_dump
        { params: @params,
          components: @components,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @components = obj[:components]
        @rng = obj[:rng]
        nil
      end

      private

      def partial_fit(x, update_comps = true)
        # initialize some variables.
        n_samples, n_features = x.shape
        scale = Math.sqrt(x.mean / @params[:n_components])
        @components = rand_uniform([@params[:n_components], n_features]) * scale if update_comps
        coefficients = rand_uniform([n_samples, @params[:n_components]]) * scale
        # optimization.
        @params[:max_iter].times do
          # update
          if update_comps
            nume = coefficients.transpose.dot(x)
            deno = coefficients.transpose.dot(coefficients).dot(@components) + @params[:eps]
            @components *= (nume / deno)
          end
          nume = x.dot(@components.transpose)
          deno = coefficients.dot(@components).dot(@components.transpose) + @params[:eps]
          coefficients *= (nume / deno)
          # normalize
          norm = Numo::NMath.sqrt((@components**2).sum(1)) + @params[:eps]
          @components /= norm.expand_dims(1) if update_comps
          coefficients *= norm
          # check convergence
          err = ((x - coefficients.dot(@components))**2).sum(1).mean
          break if err < @params[:tol]
        end
        coefficients
      end

      def rand_uniform(shape)
        rnd_vals = Array.new(shape.inject(:*)) { @rng.rand }
        Numo::DFloat.asarray(rnd_vals).reshape(shape[0], shape[1])
      end
    end
  end
end
