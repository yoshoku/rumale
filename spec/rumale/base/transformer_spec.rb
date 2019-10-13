# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Base::Transformer do
  let(:dummy_class) do
    class Dummy
      include Rumale::Base::Transformer
    end
    Dummy.new
  end

  it 'raises NotImplementedError when the fit method is not implemented.' do
    expect { dummy_class.fit }.to raise_error(NotImplementedError)
  end

  it 'raises NotImplementedError when the fit_transform method is not implemented.' do
    expect { dummy_class.fit_transform }.to raise_error(NotImplementedError)
  end
end
