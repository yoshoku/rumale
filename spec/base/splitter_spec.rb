require 'spec_helper'

RSpec.describe SVMKit::Base::Splitter do
  let(:dummy_class) do
    class Dummy
      include SVMKit::Base::Splitter
    end
    Dummy.new
  end

  it 'raises NotImplementedError when the split method is not implemented.' do
    expect{ dummy_class.split }.to raise_error(NotImplementedError)
  end
end
