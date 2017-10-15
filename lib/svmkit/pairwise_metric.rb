module SVMKit
  # Module for calculating pairwise distances, similarities, and kernels.
  module PairwiseMetric
    class << self
      # Calculating pairwise euclidean distances.
      #
      # @param x [NMatrix] (shape: [n_samples_x, n_features])
      # @param y [NMatrix] (shape: [n_samples_y, n_features])
      # @return [NMatrix] (shape: [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y] if y is given)
      def euclidean_distance(x, y = nil)
        y = x if y.nil?
        sum_x_vec = (x**2).sum(1)
        sum_y_vec = (y**2).sum(1)
        dot_xy_mat = x.dot(y.transpose)
        distance_matrix = dot_xy_mat * -2.0 +
                          sum_x_vec.repeat(y.shape[0], 1) +
                          sum_y_vec.transpose.repeat(x.shape[0], 0)
        distance_matrix.abs.sqrt
      end
    end
  end
end
