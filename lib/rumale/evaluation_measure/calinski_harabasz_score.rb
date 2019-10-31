# frozen_string_literal: true

require 'rumale/base/evaluator'

module Rumale
  module EvaluationMeasure
    # CalinskiHarabaszScore is a class that calculates the Calinski and Harabasz score.
    #
    # @example
    #   evaluator = Rumale::EvaluationMeasure::CalinskiHarabaszScore.new
    #   puts evaluator.score(x, predicted)
    #
    # *Reference*
    # - T. Calinski and J. Harabsz, "A dendrite method for cluster analysis," Communication in Statistics, Vol. 3 (1), pp. 1--27, 1972.
    class CalinskiHarabaszScore
      include Base::Evaluator

      # Calculates the Calinski and Harabasz score.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be used for calculating score.
      # @param y [Numo::Int32] (shape: [n_samples]) The predicted labels for each sample.
      # @return [Float] The Calinski and Harabasz score.
      def score(x, y)
        check_sample_array(x)
        check_label_array(y)
        check_sample_label_size(x, y)

        labels = y.to_a.uniq.sort
        n_clusters = labels.size
        n_dimensions = x.shape[1]

        centroids = Numo::DFloat.zeros(n_clusters, n_dimensions)

        within_group = 0.0
        n_clusters.times do |n|
          cls_samples = x[y.eq(labels[n]), true]
          cls_centroid = cls_samples.mean(0)
          centroids[n, true] = cls_centroid
          within_group += ((cls_samples - cls_centroid)**2).sum
        end

        return 1.0 if within_group.zero?

        mean_vec = x.mean(0)
        between_group = 0.0
        n_clusters.times do |n|
          sz_cluster = y.eq(labels[n]).count
          between_group += sz_cluster * ((centroids[n, true] - mean_vec)**2).sum
        end

        n_samples = x.shape[0]
        (between_group / (n_clusters - 1)) / (within_group / (n_samples - n_clusters))
      end
    end
  end
end
