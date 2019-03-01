# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::Node do
  it 'stores properties.' do
    left = described_class.new
    right = described_class.new
    root = described_class.new(depth: 1, impurity: 0.1, n_samples: 15, probs: 1.0, leaf: false, leaf_id: 0,
                               left: left, right: right, feature_id: 2, threshold: 0.5)
    expect(root.depth).to eq(1)
    expect(root.impurity).to eq(0.1)
    expect(root.n_samples).to eq(15)
    expect(root.probs).to eq(1.0)
    expect(root.leaf).to be_falsy
    expect(root.leaf_id).to eq(0)
    expect(root.left).to_not be_nil
    expect(root.right).to_not be_nil
    expect(root.left.class).to be(Rumale::Tree::Node)
    expect(root.right.class).to be(Rumale::Tree::Node)
    expect(root.feature_id).to eq(2)
    expect(root.threshold).to eq(0.5)
  end

  it 'dumps and restores itself using Marshal module.' do
    node = described_class.new
    copied = Marshal.load(Marshal.dump(node))
    expect(node.class).to eq(copied.class)
    expect(node.depth).to eq(copied.depth)
    expect(node.impurity).to eq(copied.impurity)
    expect(node.n_samples).to eq(copied.n_samples)
    expect(node.probs).to eq(copied.probs)
    expect(node.leaf).to eq(copied.leaf)
    expect(node.leaf_id).to eq(copied.leaf_id)
    expect(node.left).to eq(copied.left)
    expect(node.right).to eq(copied.right)
    expect(node.feature_id).to eq(copied.feature_id)
    expect(node.threshold).to eq(copied.threshold)
  end
end
