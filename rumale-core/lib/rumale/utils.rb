# frozen_string_literal: true

require 'numo/narray/alt'

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
      if shape.is_a?(Array)
        rnd_vals = Array.new(shape.inject(:*)) { rng.rand }
        Numo::DFloat.asarray(rnd_vals).reshape(shape[0], shape[1])
      else
        Numo::DFloat.asarray(Array.new(shape) { rng.rand })
      end
    end

    # @!visibility private
    def rand_normal(shape, rng = nil, mu = 0.0, sigma = 1.0)
      rng ||= Random.new
      a = rand_uniform(shape, rng)
      b = rand_uniform(shape, rng)
      (Numo::NMath.sqrt(Numo::NMath.log(a) * -2.0) * Numo::NMath.sin(b * 2.0 * Math::PI)) * sigma + mu
    end

    # @!visibility private
    def binarize_labels(labels)
      labels = labels.to_a if labels.is_a?(Numo::NArray)
      classes = labels.uniq.sort
      n_classes = classes.size
      n_samples = labels.size
      binarized = Numo::Int32.zeros(n_samples, n_classes)
      labels.each_with_index { |el, idx| binarized[idx, classes.index(el)] = 1 }
      binarized
    end

    # @!visibility private
    def normalize(x, norm)
      norm_vec = case norm
                 when 'l2'
                   Numo::NMath.sqrt((x**2).sum(axis: 1))
                 when 'l1'
                   x.abs.sum(axis: 1)
                 else
                   raise ArgumentError, 'given an unsupported norm type'
                 end
      norm_vec[norm_vec.eq(0)] = 1
      x / norm_vec.expand_dims(1)
    end
  end
end
