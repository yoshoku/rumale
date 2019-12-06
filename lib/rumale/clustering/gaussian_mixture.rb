# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/preprocessing/label_binarizer'

module Rumale
  module Clustering
    # GaussianMixture is a class that implements cluster analysis with gaussian mixture model.
    #
    # @example
    #   analyzer = Rumale::Clustering::GaussianMixture.new(n_clusters: 10, max_iter: 50)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    #   # If Numo::Linalg is installed, you can specify 'full' for the tyep of covariance option.
    #   require 'numo/linalg/autoloader'
    #   analyzer = Rumale::Clustering::GaussianMixture.new(n_clusters: 10, max_iter: 50, covariance_type: 'full')
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    class GaussianMixture
      include Base::BaseEstimator
      include Base::ClusterAnalyzer

      # Return the number of iterations to covergence.
      # @return [Integer]
      attr_reader :n_iter

      # Return the weight of each cluster.
      # @return [Numo::DFloat] (shape: [n_clusters])
      attr_reader :weights

      # Return the mean of each cluster.
      # @return [Numo::DFloat] (shape: [n_clusters, n_features])
      attr_reader :means

      # Return the diagonal elements of covariance matrix of each cluster.
      # @return [Numo::DFloat] (shape: [n_clusters, n_features] if 'diag', [n_clusters, n_features, n_features] if 'full')
      attr_reader :covariances

      # Create a new cluster analyzer with gaussian mixture model.
      #
      # @param n_clusters [Integer] The number of clusters.
      # @param init [String] The initialization method for centroids ('random' or 'k-means++').
      # @param covariance_type [String] The type of covariance parameter to be used ('diag' or 'full').
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param reg_covar [Float] The non-negative regularization to the diagonal of covariance.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_clusters: 8, init: 'k-means++', covariance_type: 'diag', max_iter: 50, tol: 1.0e-4, reg_covar: 1.0e-6, random_seed: nil)
        check_params_numeric(n_clusters: n_clusters, max_iter: max_iter, tol: tol)
        check_params_string(init: init)
        check_params_numeric_or_nil(random_seed: random_seed)
        check_params_positive(n_clusters: n_clusters, max_iter: max_iter)
        @params = {}
        @params[:n_clusters] = n_clusters
        @params[:init] = init == 'random' ? 'random' : 'k-means++'
        @params[:covariance_type] = covariance_type == 'full' ? 'full' : 'diag'
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:reg_covar] = reg_covar
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @n_iter = nil
        @weights = nil
        @means = nil
        @covariances = nil
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> GaussianMixture
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      # @return [GaussianMixture] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        check_enable_linalg('fit')

        n_samples = x.shape[0]
        memberships = init_memberships(x)
        @params[:max_iter].times do |t|
          @n_iter = t
          @weights = calc_weights(n_samples, memberships)
          @means = calc_means(x, memberships)
          @covariances = calc_covariances(x, @means, memberships, @params[:reg_covar], @params[:covariance_type])
          new_memberships = calc_memberships(x, @weights, @means, @covariances, @params[:covariance_type])
          error = (memberships - new_memberships).abs.max
          break if error <= @params[:tol]
          memberships = new_memberships.dup
        end
        self
      end

      # Predict cluster labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the cluster label.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        check_enable_linalg('predict')

        memberships = calc_memberships(x, @weights, @means, @covariances, @params[:covariance_type])
        assign_cluster(memberships)
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = check_convert_sample_array(x)
        check_enable_linalg('fit_predict')

        fit(x).predict(x)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data.
      def marshal_dump
        { params: @params,
          n_iter: @n_iter,
          weights: @weights,
          means: @means,
          covariances: @covariances }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @n_iter = obj[:n_iter]
        @weights = obj[:weights]
        @means = obj[:means]
        @covariances = obj[:covariances]
        nil
      end

      private

      def assign_cluster(memberships)
        n_clusters = memberships.shape[1]
        memberships.max_index(axis: 1) - Numo::Int32[*0.step(memberships.size - 1, n_clusters)]
      end

      def init_memberships(x)
        kmeans = Rumale::Clustering::KMeans.new(
          n_clusters: @params[:n_clusters], init: @params[:init], max_iter: 0, random_seed: @params[:random_seed]
        )
        cluster_ids = kmeans.fit_predict(x)
        encoder = Rumale::Preprocessing::LabelBinarizer.new
        Numo::DFloat.cast(encoder.fit_transform(cluster_ids))
      end

      def calc_memberships(x, weights, means, covars, covar_type)
        n_samples = x.shape[0]
        n_clusters = means.shape[0]
        memberships = Numo::DFloat.zeros(n_samples, n_clusters)
        n_clusters.times do |n|
          centered = x - means[n, true]
          covar = covar_type == 'full' ? covars[n, true, true] : covars[n, true]
          memberships[true, n] = calc_unnormalized_membership(centered, weights[n], covar, covar_type)
        end
        memberships / memberships.sum(1).expand_dims(1)
      end

      def calc_weights(n_samples, memberships)
        memberships.sum(0) / n_samples
      end

      def calc_means(x, memberships)
        memberships.transpose.dot(x) / memberships.sum(0).expand_dims(1)
      end

      def calc_covariances(x, means, memberships, reg_cover, covar_type)
        if covar_type == 'full'
          calc_full_covariances(x, means, reg_cover, memberships)
        else
          calc_diag_covariances(x, means, reg_cover, memberships)
        end
      end

      def calc_diag_covariances(x, means, reg_cover, memberships)
        n_clusters = means.shape[0]
        diag_cov = Array.new(n_clusters) do |n|
          centered = x - means[n, true]
          memberships[true, n].dot(centered**2) / memberships[true, n].sum
        end
        Numo::DFloat.asarray(diag_cov) + reg_cover
      end

      def calc_full_covariances(x, means, reg_cover, memberships)
        n_features = x.shape[1]
        n_clusters = means.shape[0]
        cov_mats = Numo::DFloat.zeros(n_clusters, n_features, n_features)
        reg_mat = Numo::DFloat.eye(n_features) * reg_cover
        n_clusters.times do |n|
          centered = x - means[n, true]
          members = memberships[true, n]
          cov_mats[n, true, true] = reg_mat + (centered.transpose * members).dot(centered) / members.sum
        end
        cov_mats
      end

      def calc_unnormalized_membership(centered, weight, covar, covar_type)
        inv_covar = calc_inv_covariance(covar, covar_type)
        inv_sqrt_det_covar = calc_inv_sqrt_det_covariance(covar, covar_type)
        distances = if covar_type == 'full'
                      (centered.dot(inv_covar) * centered).sum(1)
                    else
                      (centered * inv_covar * centered).sum(1)
                    end
        weight * inv_sqrt_det_covar * Numo::NMath.exp(-0.5 * distances)
      end

      def calc_inv_covariance(covar, covar_type)
        if covar_type == 'full'
          Numo::Linalg.inv(covar)
        else
          1.0 / covar
        end
      end

      def calc_inv_sqrt_det_covariance(covar, covar_type)
        if covar_type == 'full'
          1.0 / Math.sqrt(Numo::Linalg.det(covar))
        else
          1.0 / Math.sqrt(covar.prod)
        end
      end

      def check_enable_linalg(method_name)
        return unless @params[:covariance_type] == 'full' && !enable_linalg?
        raise "GaussianMixture##{method_name} requires Numo::Linalg when covariance_type is 'full' but that is not loaded."
      end
    end
  end
end
