# frozen_string_literal: true

module Rumale
  # This module consists of basic mix-in classes.
  module Base
    # Base module for all estimators in Rumale.
    module BaseEstimator
      # Return parameters about an estimator.
      # @return [Hash]
      attr_reader :params

      private

      def enable_parallel?
        return false if @params[:n_jobs].nil? || defined?(Parallel).nil?
        true
      end

      def n_processes
        return 1 unless enable_parallel?
        @params[:n_jobs] <= 0 ? Parallel.processor_count : @params[:n_jobs]
      end

      def parallel_map(n_outputs, &block)
        Parallel.map(Array.new(n_outputs) { |v| v }, in_processes: n_processes, &block)
      end
    end
  end
end
