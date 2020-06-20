# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'
require 'rumale/preprocessing/l1_normalizer'
require 'rumale/preprocessing/l2_normalizer'

module Rumale
  module FeatureExtraction
    # Transform sample matrix with term frequecy (tf) to a normalized tf-idf (inverse document frequency) reprensentation.
    #
    # @example
    #   encoder = Rumale::FeatureExtraction::HashVectorizer.new
    #   x = encoder.fit_transform([
    #     { foo: 1, bar: 2 },
    #     { foo: 3, baz: 1 }
    #   ])
    #
    #   # > pp x
    #   # Numo::DFloat#shape=[2,3]
    #   # [[2, 0, 1],
    #   #  [0, 1, 3]]
    #
    #   transformer = Rumale::FeatureExtraction::TfidfTransformer.new
    #   x_tfidf = transformer.fit_transform(x)
    #
    #   # > pp x_tfidf
    #   # Numo::DFloat#shape=[2,3]
    #   # [[0.959056, 0, 0.283217],
    #   #  [0, 0.491506, 0.870874]]
    #
    # *Reference*
    # - Manning, C D., Raghavan, P., and Schutze, H., "Introduction to Information Retrieval," Cambridge University Press., 2008.
    class TfidfTransformer
      include Base::BaseEstimator
      include Base::Transformer

      # Return the vector consists of inverse document frequency.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :idf

      # Create a new transfomer for converting tf vectors to tf-idf vectors.
      #
      # @param norm [String] The normalization method to be used ('l1', 'l2' and 'none').
      # @param use_idf [Boolean] The flag indicating whether to use inverse document frequency weighting.
      # @param smooth_idf [Boolean] The flag indicating whether to apply idf smoothing by log((n_samples + 1) / (df + 1)) + 1.
      # @param sublinear_tf [Boolean] The flag indicating whether to perform subliner tf scaling by 1 + log(tf).
      def initialize(norm: 'l2', use_idf: true, smooth_idf: false, sublinear_tf: false)
        check_params_string(norm: norm)
        check_params_boolean(use_idf: use_idf, smooth_idf: smooth_idf, sublinear_tf: sublinear_tf)
        @params = {}
        @params[:norm] = norm
        @params[:use_idf] = use_idf
        @params[:smooth_idf] = smooth_idf
        @params[:sublinear_tf] = sublinear_tf
        @idf = nil
      end

      # Calculate the inverse document frequency for weighting.
      #
      # @overload fit(x) -> TfidfTransformer
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate the idf values.
      # @return [TfidfTransformer]
      def fit(x, _y = nil)
        return self unless @params[:use_idf]

        x = check_convert_sample_array(x)

        n_samples = x.shape[0]
        df = x.class.cast(x.gt(0.0).count(0))

        if @params[:smooth_idf]
          df += 1
          n_samples += 1
        end

        @idf = Numo::NMath.log(n_samples / df) + 1

        self
      end

      # Calculate the idf values, and then transfrom samples to the tf-idf representation.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to calculate idf and be transformed to tf-idf representation.
      # @return [Numo::DFloat] The transformed samples.
      def fit_transform(x, _y = nil)
        fit(x).transform(x)
      end

      # Perform transforming the given samples to the tf-idf representation.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be transformed.
      # @return [Numo::DFloat] The transformed samples.
      def transform(x)
        x = check_convert_sample_array(x)
        z = x.dup

        z[z.ne(0)] = Numo::NMath.log(z[z.ne(0)]) + 1 if @params[:sublinear_tf]
        z *= @idf if @params[:use_idf]
        case @params[:norm]
        when 'l2'
          z = Rumale::Preprocessing::L2Normalizer.new.fit_transform(z)
        when 'l1'
          z = Rumale::Preprocessing::L1Normalizer.new.fit_transform(z)
        end
        z
      end
    end
  end
end
