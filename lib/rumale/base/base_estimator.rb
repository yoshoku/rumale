# frozen_string_literal: true

module Rumale
  # This module consists of basic mix-in classes.
  module Base
    # Base module for all estimators in Rumale.
    module BaseEstimator
      # Return parameters about an estimator.
      # @return [Hash]
      attr_reader :params
    end
  end
end
