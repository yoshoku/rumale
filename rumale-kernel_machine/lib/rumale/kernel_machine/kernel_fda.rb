# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/validation'

module Rumale
  module KernelMachine
    # KernelFDA is a class that implements Kernel Fisher Discriminant Analysis.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/pairwise_metric'
    #   require 'rumale/kernel_machine/kernel_fda'
    #
    #   kernel_mat_train = Rumale::PairwiseMetric::rbf_kernel(x_train)
    #   kfda = Rumale::KernelMachine::KernelFDA.new
    #   mapped_traininig_samples = kfda.fit_transform(kernel_mat_train, y)
    #
    #   kernel_mat_test = Rumale::PairwiseMetric::rbf_kernel(x_test, x_train)
    #   mapped_test_samples = kfda.transform(kernel_mat_test)
    #
    # *Reference*
    # - Baudat, G., and Anouar, F., "Generalized Discriminant Analysis using a Kernel Approach," Neural Computation, vol. 12, pp. 2385--2404, 2000.
    class KernelFDA < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Returns the eigenvectors for embedding.
      # @return [Numo::DFloat] (shape: [n_training_sampes, n_components])
      attr_reader :alphas

      # Create a new transformer with Kernel FDA.
      #
      # @param n_components [Integer] The number of components.
      # @param reg_param [Float] The regularization parameter.
      def initialize(n_components: nil, reg_param: 1e-8)
        super()
        @params = {
          n_components: n_components,
          reg_param: reg_param
        }
      end

      # Fit the model with given training data.
      # To execute this method, Numo::Linalg must be loaded.
      #
      # @param x [Numo::DFloat] (shape: [n_training_samples, n_training_samples])
      #   The kernel matrix of the training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [KernelFDA] The learned transformer itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        raise ArgumentError, 'Expect the kernel matrix of training data to be square.' unless x.shape[0] == x.shape[1]
        raise 'KernelFDA#fit requires Numo::Linalg but that is not loaded.' unless enable_linalg?(warning: false)

        # initialize some variables.
        n_samples = x.shape[0]
        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_classes = @classes.size
        n_components = if @params[:n_components].nil?
                         [n_samples, n_classes - 1].min
                       else
                         [n_samples, @params[:n_components]].min
                       end

        # centering
        @row_mean = x.mean(0)
        @all_mean = @row_mean.sum.fdiv(n_samples)
        centered_kernel_mat = x - x.mean(1).expand_dims(1) - @row_mean + @all_mean

        # calculate between and within scatter matrix.
        class_mat = Numo::DFloat.zeros(n_samples, n_samples)
        @classes.each do |label|
          idx_vec = y.eq(label)
          class_mat += Numo::DFloat.cast(idx_vec).outer(idx_vec) / idx_vec.count
        end
        between_mat = centered_kernel_mat.dot(class_mat).dot(centered_kernel_mat.transpose)
        within_mat = centered_kernel_mat.dot(centered_kernel_mat.transpose) + @params[:reg_param] * Numo::DFloat.eye(n_samples)

        # calculate projection matrix.
        _, eig_vecs = Numo::Linalg.eigh(
          between_mat, within_mat,
          vals_range: (n_samples - n_components)...n_samples
        )
        @alphas = eig_vecs.reverse(1).dup
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      # To execute this method, Numo::Linalg must be loaded.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_samples])
      #   The kernel matrix of the training data to be used for fitting the model and transformed.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_testing_samples, n_training_samples])
      #   The kernel matrix between testing samples and training samples to be transformed.
      # @return [Numo::DFloat] (shape: [n_testing_samples, n_components]) The transformed data.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        col_mean = x.sum(axis: 1) / @row_mean.shape[0]
        centered_kernel_mat = x - col_mean.expand_dims(1) - @row_mean + @all_mean
        transformed = centered_kernel_mat.dot(@alphas)
        @params[:n_components] == 1 ? transformed[true, 0].dup : transformed
      end
    end
  end
end
