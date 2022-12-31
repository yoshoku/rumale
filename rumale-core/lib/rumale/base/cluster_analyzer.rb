# frozen_string_literal: true

require 'numo/narray'

module Rumale
  module Base
    # Module for all clustering algorithms in Rumale.
    module ClusterAnalyzer
      # An abstract method for analyzing clusters and predicting cluster indices.
      def fit_predict
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # Calculate purity of clustering result.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) Testing data.
      # @param y [Numo::Int32] (shape: [n_samples]) True labels for testing data.
      # @return [Float] Purity
      def score(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_label_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        predicted = fit_predict(x)
        cluster_ids = predicted.to_a.uniq
        class_ids = y.to_a.uniq
        cluster_ids.sum do |k|
          pr_sample_ids = predicted.eq(k).where.to_a
          class_ids.map { |j| (pr_sample_ids & y.eq(j).where.to_a).size }.max
        end.fdiv(y.size)
      end
    end
  end
end
