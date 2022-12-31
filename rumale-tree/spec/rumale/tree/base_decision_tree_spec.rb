# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Tree::BaseDecisionTree do
  let(:estimator) do
    described_class.new(criterion: 'foo',
                        max_depth: 5, max_leaf_nodes: 4, min_samples_leaf: 3, max_features: 2,
                        random_seed: 1)
  end

  it 'raises NotImplementedError when calls stop_growing? method' do
    expect { estimator.send(:stop_growing?, nil) }.to raise_error(NotImplementedError)
  end

  it 'raises NotImplementedError when calls put_leaf method' do
    expect { estimator.send(:put_leaf, nil, nil) }.to raise_error(NotImplementedError)
  end

  it 'raises NotImplementedError when calls best_split method' do
    expect { estimator.send(:best_split, nil, nil, nil) }.to raise_error(NotImplementedError)
  end

  it 'raises NotImplementedError when calls impurity method' do
    expect { estimator.send(:impurity, nil) }.to raise_error(NotImplementedError)
  end

  it 'initializes some parameters', :aggregate_failures do
    expect(estimator.params[:criterion]).to eq('foo')
    expect(estimator.params[:max_depth]).to eq(5)
    expect(estimator.params[:max_leaf_nodes]).to eq(4)
    expect(estimator.params[:min_samples_leaf]).to eq(3)
    expect(estimator.params[:max_features]).to eq(2)
    expect(estimator.params[:random_seed]).to eq(1)
  end
end
