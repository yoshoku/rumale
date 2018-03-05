# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::Base::Evaluator do
  let(:dummy_class) do
    class Dummy
      include SVMKit::Base::Evaluator
    end
    Dummy.new
  end

  it 'raises NotImplementedError when the split method is not implemented.' do
    expect { dummy_class.score }.to raise_error(NotImplementedError)
  end
end
