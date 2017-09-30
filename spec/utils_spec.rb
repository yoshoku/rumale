require 'spec_helper'

RSpec.describe SVMKit::Utils do
  it 'dumps and restores NMatrix object.' do
    mat = NMatrix.rand([3, 3])
    dumped = described_class.dump_nmatrix(mat)
    restored = described_class.restore_nmatrix(dumped)
    expect(mat).to eq(restored)
  end
end
