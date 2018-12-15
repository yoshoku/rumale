# frozen_string_literal: true

module SVMKit
  # @!visibility private
  module Values
    module_function

    # @!visibility private
    def int_max
      @int_max ||= 2**([42].pack('i').size * 16 - 2) - 1
    end
  end
end
