# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'

module Rumale
  # This module consists of the classes that extract features from raw data.
  module FeatureExtraction
    # Encode array of feature-value hash to vectors.
    # This encoder turns array of mappings (Array<Hash>) with pairs of feature names and values into Numo::NArray.
    #
    # @example
    #   encoder = Rumale::FeatureExtraction::HashVectorizer.new
    #   x = encoder.fit_transform([
    #     { foo: 1, bar: 2 },
    #     { foo: 3, baz: 1 }
    #   ])
    #   # > pp x
    #   # Numo::DFloat#shape=[2,3]
    #   # [[2, 0, 1],
    #   #  [0, 1, 3]]
    #
    #   x = encoder.fit_transform([
    #     { city: 'Dubai',  temperature: 33 },
    #     { city: 'London', temperature: 12 },
    #     { city: 'San Francisco', temperature: 18 }
    #   ])
    #   # > pp x
    #   # Numo::DFloat#shape=[3,4]
    #   # [[1, 0, 0, 33],
    #   #  [0, 1, 0, 12],
    #   #  [0, 0, 1, 18]]
    #   # > pp encoder.inverse_transform(x)
    #   # [{:city=>"Dubai", :temperature=>33.0},
    #   #  {:city=>"London", :temperature=>12.0},
    #   #  {:city=>"San Francisco", :temperature=>18.0}]
    class HashVectorizer
      include Base::BaseEstimator
      include Base::Transformer

      # Return the list of feature names.
      # @return [Array] (size: [n_features])
      attr_reader :feature_names

      # Return the hash consisting of pairs of feature names and indices.
      # @return [Hash] (size: [n_features])
      attr_reader :vocabulary

      # Create a new encoder for converting array of hash consisting of feature names and values to vectors.
      #
      # @param separator [String] The separator string used for constructing new feature names for categorical feature.
      # @param sort [Boolean] The flag indicating whether to sort feature names.
      def initialize(separator: '=', sort: true)
        check_params_string(separator: separator)
        check_params_boolean(sort: sort)
        @params = {}
        @params[:separator] = separator
        @params[:sort] = sort
      end

      # Fit the encoder with given training data.
      #
      # @overload fit(x) -> HashVectorizer
      #   @param x [Array<Hash>] (shape: [n_samples]) The array of hash consisting of feature names and values.
      #   @return [HashVectorizer]
      def fit(x, _y = nil)
        @feature_names = []
        @vocabulary = {}

        x.each do |f|
          f.each do |k, v|
            k = "#{k}#{separator}#{v}".to_sym if v.is_a?(String)
            next if @vocabulary.key?(k)
            @feature_names.push(k)
            @vocabulary[k] = @vocabulary.size
          end
        end

        if sort_feature?
          @feature_names.sort!
          @feature_names.each_with_index { |k, i| @vocabulary[k] = i }
        end

        self
      end

      # Fit the encoder with given training data, then return encoded data.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Array<Hash>] (shape: [n_samples]) The array of hash consisting of feature names and values.
      #   @return [Numo::DFloat] (shape: [n_samples, n_features]) The encoded sample array.
      def fit_transform(x, _y = nil)
        fit(x).transform(x)
      end

      # Encode given the array of feature-value hash.
      #
      # @param x [Array<Hash>] (shape: [n_samples]) The array of hash consisting of feature names and values.
      # @return [Numo::DFloat] (shape: [n_samples, n_features]) The encoded sample array.
      def transform(x)
        x = [x] unless x.is_a?(Array)
        n_samples = x.size
        n_features = @vocabulary.size
        z = Numo::DFloat.zeros(n_samples, n_features)

        x.each_with_index do |f, i|
          f.each do |k, v|
            if v.is_a?(String)
              k = "#{k}#{separator}#{v}".to_sym
              v = 1
            end
            z[i, @vocabulary[k]] = v if @vocabulary.key?(k)
          end
        end

        z
      end

      # Decode sample matirx to the array of feature-value hash.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The encoded sample array.
      # @return [Array<Hash>] The array of hash consisting of feature names and values.
      def inverse_transform(x)
        n_samples = x.shape[0]
        reconst = []

        n_samples.times do |i|
          f = {}
          x[i, true].each_with_index do |el, j|
            feature_key_val(@feature_names[j], el).tap { |k, v| f[k.to_sym] = v } unless el.zero?
          end
          reconst.push(f)
        end

        reconst
      end

      private

      def feature_key_val(fname, fval)
        f = fname.to_s.split(separator)
        f.size == 2 ? f : [fname, fval]
      end

      def separator
        @params[:separator]
      end

      def sort_feature?
        @params[:sort]
      end
    end
  end
end
