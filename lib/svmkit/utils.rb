# frozen_string_literal: true

module SVMKit
  # @!visibility private
  module Utils
    module_function

    # @!visibility private
    def choice_ids(size, probs, rng = nil)
      rng ||= Random.new
      Array.new(size) do
        target = rng.rand
        chosen = 0
        probs.each_with_index do |p, idx|
          break (chosen = idx) if target <= p
          target -= p
        end
        chosen
      end
    end
  end
end
