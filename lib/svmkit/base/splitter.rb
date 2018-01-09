
module SVMKit
  module Base
    # Module for all validation methods in SVMKit.
    module Splitter
      # Return the number of splits.
      # @return [Integer]
      attr_reader :n_splits

      # An abstract method for splitting dataset.
      def split
        raise NoImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end
    end
  end
end
