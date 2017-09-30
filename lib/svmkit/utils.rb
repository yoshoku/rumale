module SVMKit
  # Module for utility methods.
  module Utils
    class << self
      # Dump an NMatrix object converted to a Ruby Hash.
      # # call-seq:
      #   dump_nmatrix(mat) -> Hash
      #
      # * *Arguments* :
      #   - +mat+ -- An NMatrix object converted to a Ruby Hash.
      # * *Returns* :
      #   - A Ruby Hash containing matrix information.
      def dump_nmatrix(mat)
        return nil if mat.class != NMatrix
        { shape: mat.shape, array: mat.to_flat_a, dtype: mat.dtype, stype: mat.stype }
      end

      # Return the results of converting the dumped data into an NMatrix object.
      #
      # call-seq:
      #   restore_nmatrix(dumped_mat) -> NMatrix
      #
      # * *Arguments* :
      #   - +dumpted_mat+ -- A Ruby Hash about NMatrix object created with SVMKit::Utils.dump_nmatrix method.
      # * *Returns* :
      #   - An NMatrix object restored from the given Hash.
      def restore_nmatrix(dmp = {})
        return nil unless dmp.class == Hash && %i[shape array dtype stype].all?(&dmp.method(:has_key?))
        NMatrix.new(dmp[:shape], dmp[:array], dtype: dmp[:dtype], stype: dmp[:stype])
      end
    end
  end
end
