# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::ExtraTreeRegressor do
  subject(:score) { estimator.score(x_test, y_test) }

  let(:dataset) { housing }
  let(:x_train) { dataset[:x_train] }
  let(:y_train) { dataset[:y_train] }
  let(:x_test) { dataset[:x_test] }
  let(:y_test) { dataset[:y_test] }
  let(:estimator) { described_class.new(min_samples_leaf: 4, random_seed: 38).fit(x_train, y_train) }

  it 'obtains high R2 score with housing dataset' do
    expect(score).to be >= 0.8
  end
end
