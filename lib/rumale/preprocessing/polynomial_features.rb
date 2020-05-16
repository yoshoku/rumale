# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  module Preprocessing
    # Generating polynomial features from the given samples.
    #
    # @example
    #   require 'rumale'
    #
    #   transformer = Rumale::Preprocessing::PolynomialFeatures.new(degree: 2)
    #   x = Numo::DFloat[[0, 1], [2, 3], [4, 5]]
    #   z = transformer.fit_transform(x)
    #   p z
    #
    #   # Numo::DFloat#shape=[3,6]
    #   # [[1, 0, 1, 0, 0, 1],
    #   #  [1, 2, 3, 4, 6, 9],
    #   #  [1, 4, 5, 16, 20, 25]]
    #
    #   # If you want to perform polynomial regression, combine it with LinearRegression as follows:
    #   ply = Rumale::Preprocessing::PolynomialFeatures.new(degree: 2)
    #   reg = Rumale::LinearModel::LinearRegression.new(fit_bias: false, random_seed: 1)
    #   pipeline = Rumale::Pipeline::Pipeline.new(steps: { trs: ply, est: reg })
    #   pipeline.fit(training_samples, training_values)
    #   results = pipeline.predict(testing_samples)
    #
    class PolynomialFeatures
      include Base::BaseEstimator
      include Base::Transformer

      # Return the number of polynomial features.
      # @return [Integer]
      attr_reader :n_output_features

      # Create a transformer for generating polynomial features.
      #
      # @param degree [Integer] The degree of polynomial features.
      def initialize(degree: 2)
        check_params_numeric(degree: degree)
        raise ArgumentError, 'Expect the value of degree parameter greater than or eqaul to 1.' if degree < 1

        @params = {}
        @params[:degree] = degree
        @n_output_features = nil
      end

      # Calculate the number of output polynomial fetures.
      #
      # @overload fit(x) -> PolynomialFeatures
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the number of output polynomial fetures.
      # @return [PolynomialFeatures]
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        n_features = x.shape[1]
        @n_output_features = 1
        @params[:degree].times do |t|
          @n_output_features += Array.new(n_features) { |n| n }.repeated_combination(t + 1).size
        end
        self
      end

      # Calculate the number of polynomial features, and then transform samples to polynomial features.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the number of polynomial features
      #     and be transformed.
      # @return [Numo::DFloat] (shape: [n_samples, n_output_features]) The transformed samples.
      def fit_transform(x, _y = nil)
        x = check_convert_sample_array(x)
        fit(x).transform(x)
      end

      # Transform the given samples to polynomial features.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be transformed.
      # @return [Numo::DFloat] (shape: [n_samples, n_output_features]) The transformed samples.
      def transform(x)
        x = check_convert_sample_array(x)
        # initialize transformed features
        n_samples, n_features = x.shape
        z = Numo::DFloat.zeros(n_samples, n_output_features)
        # bias
        z[true, 0] = 1
        curr_col = 1
        # itself
        z[true, 1..n_features] = x
        curr_col += n_features
        # high degree features
        curr_feat_ids = Array.new(n_features + 1) { |n| n + 1 }
        (1...@params[:degree]).each do
          next_feat_ids = []
          n_features.times do |d|
            f_range = curr_feat_ids[d]...curr_feat_ids.last
            next_col = curr_col + f_range.size
            z[true, curr_col...next_col] = z[true, f_range] * x[true, d..d]
            next_feat_ids.push(curr_col)
            curr_col = next_col
          end
          next_feat_ids.push(curr_col)
          curr_feat_ids = next_feat_ids
        end
        z
      end
    end
  end
end
