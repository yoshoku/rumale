# frozen_string_literal: true

require 'svmkit/validation'

module SVMKit
  module Base
    # Module for all transfomers in SVMKit.
    module Transformer
      include Validation

      # An abstract method for fitting a model.
      def fit
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      # An abstract method for fitting a model and transforming given data.
      def fit_transform
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end
    end
  end
end
