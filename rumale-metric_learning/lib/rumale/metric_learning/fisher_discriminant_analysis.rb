# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/validation'

module Rumale
  module MetricLearning
    # FisherDiscriminantAnalysis is a class that implements Fisher Discriminant Analysis.
    #
    # @example
    #   require 'rumale/metric_learning/fisher_discriminant_analysis'
    #
    #   transformer = Rumale::MetricLearning::FisherDiscriminantAnalysis.new
    #   transformer.fit(training_samples, traininig_labels)
    #   low_samples = transformer.transform(testing_samples)
    #
    # *Reference*
    # - Fisher, R. A., "The use of multiple measurements in taxonomic problems," Annals of Eugenics, vol. 7, pp. 179--188, 1936.
    # - Sugiyama, M., "Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction," Proc. ICML'06, pp. 905--912, 2006.
    class FisherDiscriminantAnalysis < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Returns the transform matrix.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Returns the mean vector.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :mean

      # Returns the class mean vectors.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :class_means

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Create a new transformer with FisherDiscriminantAnalysis.
      #
      # @param n_components [Integer] The number of components.
      #   If nil is given, the number of components will be set to [n_features, n_classes - 1].min
      def initialize(n_components: nil)
        super()
        @params = { n_components: n_components }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [FisherDiscriminantAnalysis] The learned classifier itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)
        unless enable_linalg?(warning: false)
          raise 'FisherDiscriminatAnalysis#fit requires Numo::Linalg but that is not loaded.'
        end

        # initialize some variables.
        n_features = x.shape[1]
        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_classes = @classes.size
        n_components = if @params[:n_components].nil?
                         [n_features, n_classes - 1].min
                       else
                         [n_features, @params[:n_components]].min
                       end

        # calculate within and between scatter matricies.
        within_mat = Numo::DFloat.zeros(n_features, n_features)
        between_mat = Numo::DFloat.zeros(n_features, n_features)
        @class_means = Numo::DFloat.zeros(n_classes, n_features)
        @mean = x.mean(0)
        @classes.each_with_index do |label, i|
          mask_vec = y.eq(label)
          sz_class = mask_vec.count
          class_samples = x[mask_vec, true]
          class_mean = class_samples.mean(0)
          within_mat += (class_samples - class_mean).transpose.dot(class_samples - class_mean)
          between_mat += sz_class * (class_mean - @mean).expand_dims(1) * (class_mean - @mean)
          @class_means[i, true] = class_mean
        end

        # calculate components.
        _, evecs = Numo::Linalg.eigh(between_mat, within_mat, vals_range: (n_features - n_components)...n_features)
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
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        x.dot(@components.transpose)
      end
    end
  end
end
