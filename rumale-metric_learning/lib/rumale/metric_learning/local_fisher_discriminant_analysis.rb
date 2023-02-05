# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/validation'

module Rumale
  module MetricLearning
    # LocalFisherDiscriminantAnalysis is a class that implements Local Fisher Discriminant Analysis.
    #
    # @example
    #   require 'rumale/metric_learning/local_fisher_discriminant_analysis'
    #
    #   transformer = Rumale::MetricLearning::LocalFisherDiscriminantAnalysis.new
    #   transformer.fit(training_samples, traininig_labels)
    #   low_samples = transformer.transform(testing_samples)
    #
    # *Reference*
    # - Sugiyama, M., "Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction," Proc. ICML'06, pp. 905--912, 2006.
    class LocalFisherDiscriminantAnalysis < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Returns the transform matrix.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Create a new transformer with LocalFisherDiscriminantAnalysis.
      #
      # @param n_components [Integer] The number of components.
      # @param gamma [Float] The parameter of rbf kernel, if nil it is 1 / n_features.
      def initialize(n_components: nil, gamma: nil)
        super()
        @params = {
          n_components: n_components,
          gamma: gamma
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [LocalFisherDiscriminantAnalysis] The learned classifier itself.
      def fit(x, y)
        unless enable_linalg?(warning: false)
          raise 'LocalFisherDiscriminatAnalysis#fit requires Numo::Linalg but that is not loaded.'
        end

        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_label_array(y)
        Rumale::Validation.check_sample_size(x, y)

        # initialize some variables.
        n_samples, n_features = x.shape
        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_components = @params[:n_components] || n_features
        @params[:gamma] ||= 1.fdiv(n_features)
        affinity_mat = Rumale::PairwiseMetric.rbf_kernel(x, nil, @params[:gamma])
        affinity_mat[affinity_mat.diag_indices] = 1.0

        # calculate within and mixture scatter matricies.
        class_mat = Numo::DFloat.zeros(n_samples, n_samples)
        within_weight_mat = Numo::DFloat.zeros(n_samples, n_samples)
        @classes.each do |label|
          pos = y.eq(label)
          n_class_samples = pos.count
          pos_vec = Numo::DFloat.cast(pos)
          pos_mat = pos_vec.outer(pos_vec)
          class_mat += pos_mat
          within_weight_mat += pos_mat * 1.fdiv(n_class_samples)
        end

        mixture_weight_mat = ((affinity_mat - 1) / n_samples) * class_mat + 1.fdiv(n_samples)
        within_weight_mat *= affinity_mat
        mixture_weight_mat = mixture_weight_mat.sum(axis: 1).diag - mixture_weight_mat
        within_weight_mat = within_weight_mat.sum(axis: 1).diag - within_weight_mat

        # calculate components.
        mixture_mat = x.transpose.dot(mixture_weight_mat.dot(x))
        within_mat = x.transpose.dot(within_weight_mat.dot(x))
        _, evecs = Numo::Linalg.eigh(mixture_mat, within_mat, vals_range: (n_features - n_components)...n_features)
        comps = evecs.reverse(1).transpose.dup
        @components = n_components == 1 ? comps[0, true].dup : comps.dup
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_label_array(y)
        Rumale::Validation.check_sample_size(x, y)

        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = Rumale::Validation.check_convert_sample_array(x)

        x.dot(@components.transpose)
      end
    end
  end
end
