# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Discretizes features with a given number of bins.
    # In some cases, discretizing features may accelerate decision tree training.
    #
    # @example
    #   discretizer = Rumale::Preprocessing::BinDiscretizer.new(n_bins: 4)
    #   samples = Numo::DFloat.new(5, 2).rand - 0.5
    #   transformed = discretizer.fit_transform(samples)
    #   # > pp samples
    #   # Numo::DFloat#shape=[5,2]
    #   # [[-0.438246, -0.126933],
    #   #  [ 0.294815, -0.298958],
    #   #  [-0.383959, -0.155968],
    #   #  [ 0.039948,  0.237815],
    #   #  [-0.334911, -0.449117]]
    #   # > pp transformed
    #   # Numo::DFloat#shape=[5,2]
    #   # [[0, 1],
    #   #  [3, 0],
    #   #  [0, 1],
    #   #  [2, 3],
    #   #  [0, 0]]
    class BinDiscretizer
      include Base::BaseEstimator
      include Base::Transformer

      # Return the feature steps to be used discretizing.
      # @return [Array<Numo::DFloat>] (shape: [n_features, n_bins])
      attr_reader :feature_steps

      # Create a new discretizer for features with given number of bins.
      #
      # @param n_bins [Integer] The number of bins to be used disretizing feature values.
      def initialize(n_bins: 32)
        @params = {}
        @params[:n_bins] = n_bins
        @feature_steps = nil
      end

      # Fit feature ranges to be discretized.
      #
      # @overload fit(x) -> BinDiscretizer
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the feature ranges.
      # @return [BinDiscretizer]
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        n_features = x.shape[1]
        max_vals = x.max(0)
        min_vals = x.min(0)
        @feature_steps = Array.new(n_features) do |n|
          Numo::DFloat.linspace(min_vals[n], max_vals[n], @params[:n_bins] + 1)[0...@params[:n_bins]]
        end
        self
      end

      # Fit feature ranges to be discretized, then return discretized samples.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be discretized.
      # @return [Numo::DFloat] The discretized samples.
      def fit_transform(x, _y = nil)
        x = check_convert_sample_array(x)
        fit(x).transform(x)
      end

      # Peform discretizing the given samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be discretized.
      # @return [Numo::DFloat] The discretized samples.
      def transform(x)
        x = check_convert_sample_array(x)
        n_samples, n_features = x.shape
        transformed = Numo::DFloat.zeros(n_samples, n_features)
        n_features.times do |n|
          steps = @feature_steps[n]
          @params[:n_bins].times do |bin|
            mask = x[true, n].ge(steps[bin]).where
            transformed[mask, n] = bin
          end
        end
        transformed
      end
    end
  end
end
