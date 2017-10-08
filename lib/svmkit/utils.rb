module SVMKit
  # Module for utility methods.
  module Utils
    class << self
      # Dump an NMatrix object converted to a Ruby Hash.
      #
      # @param mat [NMatrix] An NMatrix object converted to a Ruby Hash.
      # @return [Hash] A Ruby Hash containing matrix information.
      def dump_nmatrix(mat)
        return nil if mat.class != NMatrix
        { shape: mat.shape, array: mat.to_flat_a, dtype: mat.dtype, stype: mat.stype }
      end

      # Return the results of converting the dumped data into an NMatrix object.
      #
      # @param dmp [Hash] A Ruby Hash about NMatrix object created with SVMKit::Utils.dump_nmatrix method.
      # @return [NMatrix] An NMatrix object restored from the given Hash.
      def restore_nmatrix(dmp = {})
        return nil unless dmp.class == Hash && %i[shape array dtype stype].all?(&dmp.method(:has_key?))
        NMatrix.new(dmp[:shape], dmp[:array], dtype: dmp[:dtype], stype: dmp[:stype])
      end
    end
  end
end
