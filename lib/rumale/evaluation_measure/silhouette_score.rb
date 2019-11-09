# frozen_string_literal: true

require 'rumale/base/evaluator'
require 'rumale/pairwise_metric'

module Rumale
  module EvaluationMeasure
    # SilhouetteScore is a class that calculates the Silhouette Coefficient.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::SilhouetteScore.new
    #   puts evaluator.score(x, predicted)
    #
    # *Reference*
    # - P J. Rousseuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," Journal of Computational and Applied Mathematics, Vol. 20, pp. 53--65, 1987.
    class SilhouetteScore
      include Base::Evaluator

      # Create a new evaluator that calculates the silhouette coefficient.
      #
      # @param metric [String] The metric to calculate the sihouette coefficient.
      #   If metric is 'euclidean', Euclidean distance is used for dissimilarity between sample points.
      #   If metric is 'precomputed', the score method expects to be given a distance matrix.
      def initialize(metric: 'euclidean')
        check_params_string(metric: metric)
        @metric = metric
      end

      # Calculates the silhouette coefficient.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be used for calculating score.
      # @param y [Numo::Int32] (shape: [n_samples]) The predicted labels for each sample.
      # @return [Float] The mean of silhouette coefficient.
      def score(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)

        dist_mat = @metric == 'precomputed' ? x : Rumale::PairwiseMetric.euclidean_distance(x)

        labels = y.to_a.uniq.sort
        n_clusters = labels.size
        n_samples = dist_mat.shape[0]

        intra_dists = Numo::DFloat.zeros(n_samples)
        n_clusters.times do |n|
          cls_pos = y.eq(labels[n])
          sz_cluster = cls_pos.count
          next unless sz_cluster > 1
          cls_dist_mat = dist_mat[cls_pos, cls_pos].dup
          cls_dist_mat[cls_dist_mat.diag_indices] = 0.0
          intra_dists[cls_pos] = cls_dist_mat.sum(0) / (sz_cluster - 1)
        end

        inter_dists = Numo::DFloat.zeros(n_samples) + Float::INFINITY
        n_clusters.times do |m|
          cls_pos = y.eq(labels[m])
          n_clusters.times do |n|
            next if m == n
            not_cls_pos = y.eq(labels[n])
            inter_dists[cls_pos] = Numo::DFloat.minimum(
              inter_dists[cls_pos], dist_mat[cls_pos, not_cls_pos].mean(1)
            )
          end
        end

        mask = Numo::DFloat.ones(n_samples)
        n_clusters.times do |n|
          cls_pos = y.eq(labels[n])
          mask[cls_pos] = 0 unless cls_pos.count > 1
        end

        silhouettes = mask * ((inter_dists - intra_dists) / Numo::DFloat.maximum(inter_dists, intra_dists))
        silhouettes[silhouettes.isnan] = 0.0

        silhouettes.mean
      end
    end
  end
end
