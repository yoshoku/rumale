# frozen_string_literal: true

module SVMKit
  # This module consists of basic mix-in classes.
  module Base
    # Base module for all estimators in SVMKit.
    module BaseEstimator
      # Return parameters about an estimator.
      # @return [Hash]
      attr_accessor :params
    end
  end
end
