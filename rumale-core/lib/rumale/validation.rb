# frozen_string_literal: true

module Rumale
  # @!visibility private
  module Validation
    module_function

    # @!visibility private
    def check_convert_sample_array(x)
      x = Numo::DFloat.cast(x) unless x.is_a?(Numo::DFloat)
      raise ArgumentError, 'the sample array is expected to be 2-D array' unless x.ndim == 2

      x
    end

    # @!visibility private
    def check_convert_label_array(y)
      y = Numo::Int32.cast(y) unless y.is_a?(Numo::Int32)
      raise ArgumentError, 'the label array is expected to be 1-D arrray' unless y.ndim == 1

      y
    end

    # @!visibility private
    def check_convert_target_value_array(y)
      y = Numo::DFloat.cast(y) unless y.is_a?(Numo::DFloat)
      raise ArgumentError, 'the target value array is expected to be 1-D or 2-D arrray' unless [1, 2].include?(y.ndim)

      y
    end

    # @!visibility private
    def check_sample_size(x, y)
      return if x.shape[0] == y.shape[0]

      raise ArgumentError, 'the sample array and label or target value array are expected to have the same number of samples'
    end
  end
end
