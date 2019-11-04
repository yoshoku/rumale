# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::PolynomialModel::BaseFactorizationMachine do
  let(:x) { two_clusters_dataset[0] }
  let(:y) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:estimator) { described_class.new(random_seed: 1) }

  it 'raises NotImplementedError when calls partial_fit method.' do
    expect { estimator.send(:loss_gradient, x, nil, y, nil, nil) }.to raise_error(NotImplementedError)
  end

  it 'initializes some parameters.' do
    expect(estimator.params[:n_factors]).to eq(2)
    expect(estimator.params[:loss]).to be_nil
    expect(estimator.params[:reg_param_linear]).to eq(1.0)
    expect(estimator.params[:reg_param_factor]).to eq(1.0)
    expect(estimator.params[:max_iter]).to eq(1000)
    expect(estimator.params[:batch_size]).to eq(10)
    expect(estimator.params[:optimizer]).to be_a(Rumale::Optimizer::Nadam)
    expect(estimator.params[:n_jobs]).to be_nil
    expect(estimator.params[:random_seed]).to eq(1)
  end
end
