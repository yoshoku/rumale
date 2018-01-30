
module SVMKit
  module Base
    # Module for all evaluation measures in SVMKit.
    module Evaluator
      # An abstract method for evaluation of model.
      def score
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end
    end
  end
end
