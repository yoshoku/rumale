require 'svmkit/base/base_estimator'
require 'svmkit/base/transformer'

module SVMKit
  # Module for kernel approximation algorithms.
  module KernelApproximation
    # Class for RBF kernel feature mapping.
    #
    # @example
    #   transformer = SVMKit::KernelApproximation::RBF.new(gamma: 1.0, n_coponents: 128, random_seed: 1)
    #   new_training_samples = transformer.fit_transform(training_samples)
    #   new_testing_samples = transformer.transform(testing_samples)
    #
    # *Refernce*:
    # 1. A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines," Proc. NIPS'07, pp.1177--1184, 2007.
    class RBF
      include Base::BaseEstimator
      include Base::Transformer

      # Return the random matrix for transformation.
      # @return [Numo::DFloat] (shape: [n_features, n_components])
      attr_reader :random_mat

      # Return the random vector for transformation.
      # @return [Numo::DFloat] (shape: [n_components])
      attr_reader :random_vec

      # Return the random generator for transformation.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer for mapping to RBF kernel feature space.
      #
      # @param gamma [Float] The parameter of RBF kernel: exp(-gamma * x^2).
      # @param n_components [Integer] The number of dimensions of the RBF kernel feature space.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(gamma: 1.0, n_components: 128, random_seed: nil)
        @params = {}
        @params[:gamma] = gamma
        @params[:n_components] = n_components
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @random_mat = nil
        @random_vec = nil
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> RBF
      #
      # @param x [Numo::NArray] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   This method uses only the number of features of the data.
      # @return [RBF] The learned transformer itself.
      def fit(x, _y = nil)
        n_features = x.shape[1]
        @params[:n_components] = 2 * n_features if @params[:n_components] <= 0
        @random_mat = rand_normal([n_features, @params[:n_components]]) * (2.0 * @params[:gamma])**0.5
        n_half_components = @params[:n_components] / 2
        @random_vec = Numo::DFloat.zeros(@params[:n_components] - n_half_components).concatenate(
          Numo::DFloat.ones(n_half_components) * (0.5 * Math::PI)
        )
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _y = nil)
        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @overload transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        n_samples, = x.shape
        projection = x.dot(@random_mat) + @random_vec.tile(n_samples, 1)
        Numo::NMath.sin(projection) * ((2.0 / @params[:n_components])**0.5)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about RBF.
      def marshal_dump
        { params: @params,
          random_mat: @random_mat,
          random_vec: @random_vec,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @random_mat = obj[:random_mat]
        @random_vec = obj[:random_vec]
        @rng = obj[:rng]
        nil
      end

      protected

      # Generate the uniform random matrix with the given shape.
      def rand_uniform(shape)
        rnd_vals = Array.new(shape.inject(:*)) { @rng.rand }
        Numo::DFloat.asarray(rnd_vals).reshape(shape[0], shape[1])
      end

      # Generate the normal random matrix with the given shape, mean, and standard deviation.
      def rand_normal(shape, mu = 0.0, sigma = 1.0)
        a = rand_uniform(shape)
        b = rand_uniform(shape)
        (Numo::NMath.sqrt(Numo::NMath.log(a) * -2.0) * Numo::NMath.sin(b * 2.0 * Math::PI)) * sigma + mu
      end
    end
  end
end
