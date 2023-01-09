# frozen_string_literal: true

require 'rumale/base/estimator'

module Rumale
  # This module consists of the classes that implement generalized linear models.
  module LinearModel
    # BaseEstimator is an abstract class for implementation of linear model. This class is used internally.
    class BaseEstimator < Rumale::Base::Estimator
      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_outputs/n_classes, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept).
      # @return [Numo::DFloat] (shape: [n_outputs/n_classes])
      attr_reader :bias_term

      # Create an initial linear model.

      private

      def expand_feature(x)
        n_samples = x.shape[0]
        Numo::NArray.hstack([x, Numo::DFloat.ones([n_samples, 1]) * @params[:bias_scale]])
      end

      def split_weight(w)
        if w.ndim == 1
          if fit_bias?
            [w[0...-1].dup, w[-1]]
          else
            [w, 0.0]
          end
        elsif fit_bias?
          [w[true, 0...-1].dup, w[true, -1].dup]
        else
          [w, Numo::DFloat.zeros(w.shape[0])]
        end
      end

      def fit_bias?
        @params[:fit_bias] == true
      end
    end
  end
end
