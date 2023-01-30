# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/cluster_analyzer'
require 'rumale/pairwise_metric'
require 'rumale/validation'

module Rumale
  module Clustering
    # MeanShift is a class that implements mean-shift clustering with flat kernel.
    #
    # @example
    #   require 'rumale/clustering/mean_shift'
    #
    #   analyzer = Rumale::Clustering::MeanShift.new(bandwidth: 1.5)
    #   cluster_labels = analyzer.fit_predict(samples)
    #
    # *Reference*
    # - Carreira-Perpinan, M A., "A review of mean-shift algorithms for clustering," arXiv:1503.00687v1.
    # - Sheikh, Y A., Khan, E A., and Kanade, T., "Mode-seeking by Medoidshifts," Proc. ICCV'07, pp. 1--8, 2007.
    # - Vedaldi, A., and Soatto, S., "Quick Shift and Kernel Methods for Mode Seeking," Proc. ECCV'08, pp. 705--718, 2008.
    class MeanShift < Rumale::Base::Estimator
      include Rumale::Base::ClusterAnalyzer

      # Return the centroids.
      # @return [Numo::DFloat] (shape: [n_clusters, n_features])
      attr_reader :cluster_centers

      # Create a new cluster analyzer with mean-shift algorithm.
      #
      # @param bandwidth [Float] The bandwidth parameter of flat kernel.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion
      def initialize(bandwidth: 1.0, max_iter: 500, tol: 1e-4)
        super()
        @params = {
          bandwidth: bandwidth,
          max_iter: max_iter,
          tol: tol
        }
      end

      # Analysis clusters with given training data.
      #
      # @overload fit(x) -> MeanShift
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      #   @return [MeanShift] The learned cluster analyzer itself.
      def fit(x, _y = nil)
        x = Rumale::Validation.check_convert_sample_array(x)

        z = x.dup
        @params[:max_iter].times do
          distance_mat = Rumale::PairwiseMetric.euclidean_distance(x, z)
          kernel_mat = Numo::DFloat.cast(distance_mat.le(@params[:bandwidth]))
          sum_kernel = kernel_mat.sum(axis: 0)
          weight_mat = kernel_mat.dot((1 / sum_kernel).diag)
          updated = weight_mat.transpose.dot(x)
          break if (z - updated).abs.sum(axis: 1).max <= @params[:tol]

          z = updated
        end

        @cluster_centers = connect_components(z)
        p z
        p @cluster_centers

        self
      end

      # Predict cluster labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the cluster label.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def predict(x)
        x = Rumale::Validation.check_convert_sample_array(x)

        assign_cluster(x)
      end

      # Analysis clusters and assign samples to clusters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for cluster analysis.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted cluster label per sample.
      def fit_predict(x)
        x = Rumale::Validation.check_convert_sample_array(x)

        fit(x).predict(x)
      end

      private

      def assign_cluster(x)
        n_clusters = @cluster_centers.shape[0]
        distance_mat = Rumale::PairwiseMetric.squared_error(x, @cluster_centers)
        distance_mat.min_index(axis: 1) - Numo::Int32[*0.step(distance_mat.size - 1, n_clusters)]
      end

      def connect_components(z)
        centers = []
        n_samples = z.shape[0]

        n_samples.times do |idx|
          assigned = false
          centers.each do |cluster_vec|
            dist = Math.sqrt(((z[idx, true] - cluster_vec)**2).sum.abs)
            if dist <= @params[:bandwidth]
              assigned = true
              break
            end
          end
          centers << z[idx, true].dup unless assigned
        end

        Numo::DFloat.asarray(centers)
      end
    end
  end
end
