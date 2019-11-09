# frozen_string_literal: true

require 'rumale/base/evaluator'
require 'rumale/pairwise_metric'

module Rumale
  module EvaluationMeasure
    # DaviesBouldinScore is a class that calculates the Davies-Bouldin score.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::DaviesBouldinScore.new
    #   puts evaluator.score(x, predicted)
    #
    # *Reference*
    # - D L. Davies and D W. Bouldin, "A Cluster Separation Measure," IEEE Trans. Pattern Analysis and Machine Intelligence, Vol. PAMI-1, No. 2, pp. 224--227, 1979.
    class DaviesBouldinScore
      include Base::Evaluator

      # Calculates the Davies-Bouldin score.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be used for calculating score.
      # @param y [Numo::Int32] (shape: [n_samples]) The predicted labels for each sample.
      # @return [Float] The Davies-Bouldin score.
      def score(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)

        labels = y.to_a.uniq.sort
        n_clusters = labels.size
        n_dimensions = x.shape[1]

        dist_cluster = Numo::DFloat.zeros(n_clusters)
        centroids = Numo::DFloat.zeros(n_clusters, n_dimensions)

        n_clusters.times do |n|
          cls_samples = x[y.eq(labels[n]), true]
          cls_centroid = cls_samples.mean(0)
          centroids[n, true] = cls_centroid
          dist_cluster[n] = Rumale::PairwiseMetric.euclidean_distance(cls_samples, cls_centroid.expand_dims(0)).mean
        end

        dist_centroid = Rumale::PairwiseMetric.euclidean_distance(centroids)
        # p dist_cluster
        # p dist_centroid
        dist_centroid[dist_centroid.eq(0)] = Float::INFINITY
        dist_mat = (dist_cluster.expand_dims(1) + dist_cluster) / dist_centroid
        dist_mat[dist_mat.diag_indices] = -Float::INFINITY
        dist_mat.max(0).mean
      end
    end
  end
end
