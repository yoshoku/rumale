# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::Node do
  let(:left) { described_class.new }
  let(:right) { described_class.new }
  let(:root) do
    described_class.new(depth: 1, impurity: 0.1, n_samples: 15, probs: 1.0, leaf: false, leaf_id: 0,
                        left: left, right: right, feature_id: 2, threshold: 0.5)
  end

  it 'stores properties', :aggregate_failures do
    expect(root.depth).to eq(1)
    expect(root.impurity).to eq(0.1)
    expect(root.n_samples).to eq(15)
    expect(root.probs).to eq(1.0)
    expect(root.leaf).to be_falsy
    expect(root.leaf_id).to eq(0)
    expect(root.left).not_to be_nil
    expect(root.right).not_to be_nil
    expect(root.left.class).to be(described_class)
    expect(root.right.class).to be(described_class)
    expect(root.feature_id).to eq(2)
    expect(root.threshold).to eq(0.5)
  end
end
