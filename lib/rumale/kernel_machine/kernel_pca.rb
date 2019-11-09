# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  module KernelMachine
    # KernelPCA is a class that implements Kernel Principal Component Analysis.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #
    #   kernel_mat_train = Rumale::PairwiseMetric::rbf_kernel(training_samples)
    #   kpca = Rumale::KernelMachine::KernelPCA(n_components: 2)
    #   mapped_traininig_samples = kpca.fit_transform(kernel_mat_train)
    #
    #   kernel_mat_test = Rumale::PairwiseMetric::rbf_kernel(test_samples, training_samples)
    #   mapped_test_samples = kpca.transform(kernel_mat_test)
    #
    # *Reference*
    # - B. Scholkopf, A. Smola, and K-R. Muller, "Nonlinear Component Analysis as a Kernel Eigenvalue Problem," Neural Computation, Vol. 10 (5), pp. 1299--1319, 1998.
    class KernelPCA
      include Base::BaseEstimator
      include Base::Transformer

      # Returns the eigenvalues of the centered kernel matrix.
      # @return [Numo::DFloat] (shape: [n_components])
      attr_reader :lambdas

      # Returns the eigenvectros of the centered kernel matrix.
      # @return [Numo::DFloat] (shape: [n_training_sampes, n_components])
      attr_reader :alphas

      # Create a new transformer with Kernel PCA.
      #
      # @param n_components [Integer] The number of components.
      def initialize(n_components: 2)
        check_params_integer(n_components: n_components)
        @params = {}
        @params[:n_components] = n_components
        @alphas = nil
        @lambdas = nil
        @row_mean = nil
        @all_mean = nil
      end

      # Fit the model with given training data.
      # To execute this method, Numo::Linalg must be loaded.
      #
      # @overload fit(x) -> KernelPCA
      #   @param x [Numo::DFloat] (shape: [n_training_samples, n_training_samples])
      #     The kernel matrix of the training data to be used for fitting the model.
      # @return [KernelPCA] The learned transformer itself.
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        raise ArgumentError, 'Expect the kernel matrix of training data to be square.' unless x.shape[0] == x.shape[1]
        raise 'KernelPCA#fit requires Numo::Linalg but that is not loaded.' unless enable_linalg?

        n_samples = x.shape[0]
        @row_mean = x.mean(0)
        @all_mean = @row_mean.sum.fdiv(n_samples)
        centered_kernel_mat = x - x.mean(1).expand_dims(1) - @row_mean + @all_mean
        eig_vals, eig_vecs = Numo::Linalg.eigh(centered_kernel_mat, vals_range: (n_samples - @params[:n_components])...n_samples)
        @alphas = eig_vecs.reverse(1).dup
        @lambdas = eig_vals.reverse.dup
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      # To execute this method, Numo::Linalg must be loaded.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_samples])
      #     The kernel matrix of the training data to be used for fitting the model and transformed.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _y = nil)
        x = check_convert_sample_array(x)
        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #   The kernel matrix between testing samples and training samples to be transformed.
      # @return [Numo::DFloat] (shape: [n_testing_samples, n_components]) The transformed data.
      def transform(x)
        x = check_convert_sample_array(x)
        col_mean = x.sum(1) / @row_mean.shape[0]
        centered_kernel_mat = x - col_mean.expand_dims(1) - @row_mean + @all_mean
        transform_mat = @alphas.dot((1.0 / Numo::NMath.sqrt(@lambdas)).diag)
        transformed = centered_kernel_mat.dot(transform_mat)
        @params[:n_components] == 1 ? transformed[true, 0].dup : transformed
      end

      # Dump marshal data.
      # @return [Hash] The marshal data.
      def marshal_dump
        { params: @params,
          row_mean: @row_mean,
          all_mean: @all_mean,
          alphas: @alphas,
          lambdas: @lambdas }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @row_mean = obj[:row_mean]
        @all_mean = obj[:all_mean]
        @alphas = obj[:alphas]
        @lambdas = obj[:lambdas]
        nil
      end
    end
  end
end
