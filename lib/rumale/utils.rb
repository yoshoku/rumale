# frozen_string_literal: true

module Rumale
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

    # @!visibility private
    def rand_uniform(shape, rng = nil)
      rng ||= Random.new
      rnd_vals = Array.new(shape.inject(:*)) { rng.rand }
      Numo::DFloat.asarray(rnd_vals).reshape(shape[0], shape[1])
    end

    # @!visibility private
    def rand_normal(shape, rng = nil, mu = 0.0, sigma = 1.0)
      rng ||= Random.new
      a = rand_uniform(shape, rng)
      b = rand_uniform(shape, rng)
      (Numo::NMath.sqrt(Numo::NMath.log(a) * -2.0) * Numo::NMath.sin(b * 2.0 * Math::PI)) * sigma + mu
    end
  end
end
