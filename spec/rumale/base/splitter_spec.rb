# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Base::Splitter do
  let(:dummy_class) do
    class Dummy
      include Rumale::Base::Splitter
    end
    Dummy.new
  end

  it 'raises NotImplementedError when the split method is not implemented.' do
    expect { dummy_class.split }.to raise_error(NotImplementedError)
  end
end
