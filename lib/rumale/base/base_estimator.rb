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

      def enable_linalg?(warning: true)
        if defined?(Numo::Linalg).nil?
          warn('If you want to use features that depend on Numo::Linalg, you should install and load Numo::Linalg in advance.') if warning
          return false
        end
        if Numo::Linalg::VERSION < '0.1.4'
          if warning
            warn('The loaded Numo::Linalg does not implement the methods required by Rumale. Please load Numo::Linalg version 0.1.4 or later.')
          end
          return false
        end
        true
      end

      def enable_parallel?
        return false if @params[:n_jobs].nil?

        if defined?(Parallel).nil?
          warn('If you want to use parallel option, you should install and load Parallel in advance.')
          return false
        end
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
