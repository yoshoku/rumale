# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::Base::ClusterAnalyzer do
  let(:dummy_class) do
    class Dummy
      include SVMKit::Base::ClusterAnalyzer
    end
    Dummy.new
  end

  it 'raises NotImplementedError when the fit method is not implemented.' do
    expect { dummy_class.fit_predict }.to raise_error(NotImplementedError)
  end
end
