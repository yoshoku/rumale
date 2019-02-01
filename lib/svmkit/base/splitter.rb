# frozen_string_literal: true

require 'svmkit/validation'

module SVMKit
  module Base
    # Module for all validation methods in SVMKit.
    module Splitter
      include Validation

      # Return the number of splits.
      # @return [Integer]
      attr_reader :n_splits

      # An abstract method for splitting dataset.
      def split
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end
    end
  end
end
