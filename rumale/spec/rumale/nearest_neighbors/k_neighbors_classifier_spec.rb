# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KNeighborsClassifier do
  subject(:score) { estimator.score(x_test, y_test) }

  let(:dataset) { iris }
  let(:x_train) { dataset[:x_train] }
  let(:y_train) { dataset[:y_train] }
  let(:x_test) { dataset[:x_test] }
  let(:y_test) { dataset[:y_test] }
  let(:estimator) { described_class.new.fit(x_train, y_train) }

  it 'obtains high classification accuracy with iris dataset' do
    expect(score).to be >= 0.8
  end
end
