# frozen_string_literal: true

module Rumale
  module Ensemble
    # @!visibility private
    module Value
      # @!visibility private
      N_BITS = 1.size * 8
      # @!visibility private
      SEED_BASE = 2**(N_BITS - 1) - 1
    end
  end
end
