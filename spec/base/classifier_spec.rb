# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::Base::Classifier do
  let(:dummy_class) do
    class Dummy
      include SVMKit::Base::Classifier
    end
    Dummy.new
  end

  it 'raises NotImplementedError when the fit method is not implemented.' do
    expect { dummy_class.fit }.to raise_error(NotImplementedError)
  end

  it 'raises NotImplementedError when the predict method is not implemented.' do
    expect { dummy_class.predict }.to raise_error(NotImplementedError)
  end
end
