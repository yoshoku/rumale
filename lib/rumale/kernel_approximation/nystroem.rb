# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'
require 'rumale/pairwise_metric'

module Rumale
  module KernelApproximation
    # Nystroem is a class that implements feature mapping with Nystroem method.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #
    #   transformer = Rumale::KernelApproximation::Nystroem.new(kernel: 'rbf', gamma: 1, n_components: 128, random_seed: 1)
    #   new_training_samples = transformer.fit_transform(training_samples)
    #   new_testing_samples = transformer.transform(testing_samples)
    #
    # *Reference*
    # - Yang, T., Li, Y., Mahdavi, M., Jin, R., and Zhou, Z-H., "Nystrom Method vs Random Fourier Features: A Theoretical and Empirical Comparison," Advances in NIPS'12, Vol. 1, pp. 476--484, 2012.
    class Nystroem
      include Base::BaseEstimator
      include Base::Transformer

      # Returns the randomly sampled training data for feature mapping.
      # @return [Numo::DFloat] (shape: n_components, n_features])
      attr_reader :components

      # Returns the indices sampled training data.
      # @return [Numo::Int32] (shape: [n_components])
      attr_reader :component_indices

      # Returns the normalizing factors.
      # @return [Numo::DFloat] (shape: [n_components, n_components])
      attr_reader :normalizer

      # Return the random generator for transformation.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer for mapping to kernel feature space with Nystrom method.
      #
      # @param kernel [String] The type of kernel function ('rbf', 'linear', 'poly', and 'sigmoid)
      # @param gamma [Float] The gamma parameter in rbf/poly/sigmoid kernel function.
      # @param degree [Integer] The degree parameter in polynomial kernel function.
      # @param coef [Float] The coefficient in poly/sigmoid kernel function.
      # @param n_components [Integer] The number of dimensions of the kernel feature space.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(kernel: 'rbf', gamma: 1, degree: 3, coef: 1, n_components: 100, random_seed: nil)
        check_params_string(kernel: kernel)
        check_params_numeric(gamma: gamma, coef: coef, degree: degree, n_components: n_components)
        check_params_numeric_or_nil(random_seed: random_seed)
        @params = method(:initialize).parameters.map { |_t, arg| [arg, binding.local_variable_get(arg)] }.to_h
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @component_indices = nil
        @components = nil
        @normalizer = nil
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> Nystroem
      #   @param x [Numo::NArray] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @return [Nystroem] The learned transformer itself.
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        raise 'Nystroem#fit requires Numo::Linalg but that is not loaded.' unless enable_linalg?

        # initialize some variables.
        sub_rng = @rng.dup
        n_samples = x.shape[0]
        n_components = [1, [@params[:n_components], n_samples].min].max

        # random sampling.
        @component_indices = Numo::Int32.cast(Array(0...n_samples).shuffle(random: sub_rng)[0...n_components])
        @components = x[@component_indices, true]

        # calculate normalizing factor.
        kernel_mat = kernel_mat(@components)
        eig_vals, eig_vecs = Numo::Linalg.eigh(kernel_mat)
        la = eig_vals.class.maximum(eig_vals.reverse, 1e-12)
        u = eig_vecs.reverse(1)
        @normalizer = u.dot((1.0 / Numo::NMath.sqrt(la)).diag)

        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _y = nil)
        x = check_convert_sample_array(x)
        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = check_convert_sample_array(x)
        z = kernel_mat(x, @components)
        z.dot(@normalizer)
      end

      private

      def kernel_mat(x, y = nil)
        case @params[:kernel]
        when 'rbf'
          Rumale::PairwiseMetric.rbf_kernel(x, y, @params[:gamma])
        when 'poly'
          Rumale::PairwiseMetric.polynomial_kernel(x, y, @params[:degree], @params[:gamma], @params[:coef])
        when 'sigmoid'
          Rumale::PairwiseMetric.sigmoid_kernel(x, y, @params[:gamma], @params[:coef])
        when 'linear'
          Rumale::PairwiseMetric.linear_kernel(x, y)
        else
          raise ArgumentError, "Expect kernel parameter to be given 'rbf', 'linear', 'poly', or 'sigmoid'."
        end
      end
    end
  end
end
